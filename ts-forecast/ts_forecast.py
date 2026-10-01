#!/usr/bin/env python3
"""ts_forecast.py — универсальный прогноз временных рядов (zero-shot, Chronos-Bolt, Apache-2.0).

Вход: csv/xlsx (колонка дат + одна или несколько числовых колонок) или --inline "1,2,3".
Выход: xlsx-отчёт (история + прогноз с квантилями + сводка [+ метрики бэктеста]), график png, сводка в консоль.

Примеры:
  bash scripts/run.sh --csv data.xlsx --sheet "Лист1" --date-col "Дата" \
      --value-cols "Продажи,Трафик" --freq M --horizon 12 --backtest 12 \
      --out forecast.xlsx --chart forecast.png
  bash scripts/run.sh --inline "10,12,14,13,15,17,16,18" --freq D --horizon 5
"""
import argparse, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

FREQ_ALIASES = {
    "D": "D", "дн": "D", "day": "D", "daily": "D",
    "W": "W-SUN", "нед": "W-SUN", "week": "W-SUN", "weekly": "W-SUN",
    "M": "MS", "мес": "MS", "month": "MS", "monthly": "MS",
    "Q": "QS", "кв": "QS", "quarter": "QS", "quarterly": "QS",
    "Y": "YS", "год": "YS", "year": "YS", "yearly": "YS",
}
FREQ_RU = {"D": "дни", "W-SUN": "недели", "MS": "месяцы", "QS": "кварталы", "YS": "годы", None: "индекс"}


def infer_freq(dates: pd.Series) -> str:
    dif = pd.Series(pd.to_datetime(dates)).sort_values().diff().dropna().dt.days
    m = float(dif.median()) if len(dif) else 1.0
    if m <= 1.5: return "D"
    if m <= 10: return "W-SUN"
    if m <= 45: return "MS"
    if m <= 135: return "QS"
    return "YS"


def load_input(args):
    """Возвращает (dict имя->np.array, dates pd.DatetimeIndex|None, freq_rule|None)."""
    if args.inline:
        vals = [float(x) for x in args.inline.replace(";", ",").split(",") if x.strip() != ""]
        s = pd.Series(vals, index=pd.RangeIndex(len(vals)))
        return {args.inline_label: s}, None, None

    p = Path(args.csv)
    if not p.exists():
        sys.exit(f"Файл не найден: {p}")
    if p.suffix.lower() in (".xlsx", ".xlsm"):
        df = pd.read_excel(p, sheet_name=args.sheet or 0)
    else:
        df = pd.read_csv(p, sep=None, engine="python")

    dates = None
    if args.date_col:
        raw = df[args.date_col]
        d_iso = pd.to_datetime(raw, format="%Y-%m-%d", errors="coerce")
        if d_iso.notna().mean() >= 0.9:
            dates = d_iso
        else:
            dates = pd.to_datetime(raw, dayfirst=True, errors="coerce")
        ok = dates.notna()
        df, dates = df[ok], dates[ok]

    if args.value_cols:
        cols = [c.strip() for c in args.value_cols.split(";")]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            sys.exit(f"Колонки не найдены: {missing}\nДоступные: {list(df.columns)[:20]}")
    else:
        cols = [c for c in df.columns if c != args.date_col and pd.api.types.is_numeric_dtype(df[c])]
        if not cols:
            sys.exit("Не найдено числовых колонок — укажите --value-cols")

    out = {}
    for c in cols:
        v = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        out[c] = pd.Series(v.to_numpy(float), index=dates if dates is not None else pd.RangeIndex(len(v)))
    return out, dates, None


def to_regular(s: pd.Series, rule: str, agg: str, fill: str):
    if rule is not None and isinstance(s.index, pd.DatetimeIndex):
        s = s.sort_index().resample(rule).agg(agg)
        if rule == "W-SUN":  # метки недель: воскресенье -> понедельник (бизнес-неделя пн–вс)
            s.index = s.index - pd.Timedelta(days=6)
    if fill == "interp":
        s = s.interpolate(limit_direction="both")
    elif fill == "ffill":
        s = s.ffill().bfill()
    else:
        s = s.fillna(0.0)
    return s


def future_index(index, rule, h):
    if rule is None or not isinstance(index, pd.DatetimeIndex):
        start = (index[-1] + 1) if len(index) else 1
        return pd.RangeIndex(start, start + h)
    if rule == "D":
        return pd.date_range(index[-1] + pd.Timedelta(days=1), periods=h, freq="D")
    if rule == "W-SUN":
        return pd.date_range(index[-1] + pd.Timedelta(days=7), periods=h, freq="7D")
    return pd.date_range(index[-1] + pd.Timedelta(days=1), periods=h, freq=rule)


def predict_quantiles(pipe, vals: np.ndarray, h: int):
    import torch
    ctx = torch.tensor(vals, dtype=torch.float32)
    t0 = time.time()
    raw = pipe.predict(ctx, prediction_length=h)
    q = raw[0] if getattr(raw, "ndim", 2) == 3 else raw
    q = q.detach().cpu().numpy() if hasattr(q, "detach") else np.asarray(q)
    if q.ndim == 1:
        q = q.reshape(1, -1)
    if q.shape[0] != 9 and q.shape[1] == 9:
        q = q.T
    return q, time.time() - t0


def wape(y, f):
    denom = float(np.abs(y).sum())
    return round(float(np.abs(y - f).sum() / denom), 3) if denom else None


def backtest_metrics(pipe, vals, n, ma_window):
    ctx, hold = vals[:-n], vals[-n:]
    q, _ = predict_quantiles(pipe, ctx, n)
    med = q[4] if q.shape[0] >= 5 else q[len(q) // 2]
    w = min(ma_window, len(ctx)) if ma_window else min(7, len(ctx))
    out = {"WAPE модель": wape(hold, med[:n]),
           "WAPE наивный": wape(hold, np.repeat(ctx[-1], n)),
           "WAPE MA": wape(hold, np.repeat(np.mean(ctx[-w:]), n))}
    if len(ctx) >= 14:
        lag = 7
        out["WAPE сезонный(лаг-7)"] = wape(hold, np.array([ctx[len(ctx) - lag + (i % lag)] for i in range(n)]))
    return out


def main():
    ap = argparse.ArgumentParser(description="Универсальный прогноз временных рядов (Chronos-Bolt)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="путь к csv/xlsx")
    src.add_argument("--inline", help="ряд числами через запятую")
    ap.add_argument("--inline-label", default="ряд")
    ap.add_argument("--sheet", help="лист для xlsx")
    ap.add_argument("--date-col", help="колонка с датой")
    ap.add_argument("--value-cols", help="колонки значений через «;» (в названиях бывают запятые; по умолчанию все числовые)")
    ap.add_argument("--freq", help="D/W/M/Q/Y (по умолчанию авто по датам)")
    ap.add_argument("--resample", choices=["sum", "mean"], default="sum")
    ap.add_argument("--fill", choices=["zero", "interp", "ffill"], default="zero")
    ap.add_argument("--horizon", type=int, required=True, help="горизонт в периодах частоты")
    ap.add_argument("--backtest", type=int, default=0, help="проверка на последних N точках (0 = выкл)")
    ap.add_argument("--ma-window", type=int, default=7, help="окно MA-бейзлайна")
    ap.add_argument("--floor", type=float, default=None, help="нижняя граница прогноза (напр. 0 для счётчиков)")
    ap.add_argument("--out", help="xlsx-отчёт")
    ap.add_argument("--chart", help="png-график (при нескольких рядах добавится суффикс)")
    ap.add_argument("--model", default="amazon/chronos-bolt-base", help="id модели на HF")
    args = ap.parse_args()

    series, dates, _ = load_input(args)
    if args.freq:
        rule = FREQ_ALIASES.get(args.freq, args.freq)
    else:
        rule = infer_freq(dates) if dates is not None else None

    print(f"Рядов: {len(series)} · частота: {FREQ_RU.get(rule, rule)} · горизонт: {args.horizon}")
    from chronos import BaseChronosPipeline
    t0 = time.time()
    pipe = BaseChronosPipeline.from_pretrained(args.model, device_map="cpu")
    print(f"Модель загружена за {time.time()-t0:.1f} c ({args.model})")

    summary, sheets, metrics = [], {}, []
    for name, s in series.items():
        s = to_regular(s, rule, args.resample, args.fill)
        vals = s.to_numpy(float)
        if len(vals) < 4:
            print(f"⚠️ {name}: слишком короткий ряд ({len(vals)}), пропуск")
            continue
        if len(vals) > 4000:
            print(f"⚠️ {name}: длинный ряд ({len(vals)}) — Bolt эффективен на контексте ~2–4 тыс. точек")
        q, sec = predict_quantiles(pipe, vals, args.horizon)
        if args.floor is not None:
            q = np.maximum(q, args.floor)
        med = q[4] if q.shape[0] >= 5 else q[len(q) // 2]
        fut = future_index(s.index, rule, args.horizon)

        hist = pd.DataFrame({"дата": list(s.index) if isinstance(s.index, pd.DatetimeIndex) else list(s.index),
                             "факт": vals})
        qcols = {f"p{10*(i+1)}": q[i] for i in range(q.shape[0])}
        fdf = pd.DataFrame({"дата": list(fut), **qcols, "прогноз": med})
        sheets[name] = (hist, fdf)
        row = {"Ряд": name, "Точек": len(vals),
               "Последняя точка": str(s.index[-1])[:10],
               "Прогноз (медиана) сумма": round(float(med.sum()), 1),
               "P10 сумма": round(float(q[0].sum()), 1), "P90 сумма": round(float(q[-1].sum()), 1),
               "Последнее значение": round(float(vals[-1]), 1), "infer, с": round(sec, 2)}
        if args.backtest:
            m = backtest_metrics(pipe, vals, min(args.backtest, len(vals) - 3), args.ma_window)
            row.update(m); metrics.append({"Ряд": name, **m, "Точек бэктеста": args.backtest})
            print(f"  бэктест {name}: " + " | ".join(f"{k}={v}" for k, v in m.items()))
        summary.append(row)
        print(f"✅ {name}: прогноз {args.horizon}×{FREQ_RU.get(rule,'шаг')}; медиана-сумма {row['Прогноз (медиана) сумма']}, вилка {row['P10 сумма']}…{row['P90 сумма']} ({sec:.2f} c)")

    if args.out:
        out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(out, engine="openpyxl") as xw:
            pd.DataFrame(summary).to_excel(xw, sheet_name="Сводка", index=False)
            if metrics:
                pd.DataFrame(metrics).to_excel(xw, sheet_name="Метрики", index=False)
            used = set()
            for name, (hist, fdf) in sheets.items():
                sh = name[:28].replace("/", "_")
                base, i = sh, 1
                while sh in used: i += 1; sh = f"{base[:26]}_{i}"
                used.add(sh)
                pd.concat([hist.assign(тип="факт"), fdf.assign(тип="прогноз")], ignore_index=True).to_excel(xw, sheet_name=sh, index=False)
        print(f"📊 xlsx: {out}")

    if args.chart and sheets:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        multi = len(sheets) > 1
        for i, (name, (hist, fdf)) in enumerate(sheets.items()):
            fig, ax = plt.subplots(figsize=(11, 4.5), dpi=150)
            tail = hist.tail(180)
            ax.plot(pd.to_datetime(tail["дата"]) if isinstance(tail["дата"].iloc[0], (str, pd.Timestamp)) else tail["дата"],
                    tail["факт"], label="факт", color="#1f3864", lw=1.6)
            fx = pd.to_datetime(fdf["дата"]) if isinstance(fdf["дата"].iloc[0], (str, pd.Timestamp)) else fdf["дата"]
            ax.plot(fx, fdf["прогноз"], label="прогноз (медиана)", color="#2e75b6", lw=2)
            if "p10" in fdf and "p90" in fdf:
                ax.fill_between(fx, fdf["p10"], fdf["p90"], color="#2e75b6", alpha=0.18, label="вилка p10–p90")
            ax.set_title(f"Прогноз: {name}")
            ax.grid(alpha=0.25); ax.legend(loc="best"); fig.autofmt_xdate()
            path = Path(args.chart)
            if multi:
                path = path.with_name(f"{path.stem}_{i+1}{path.suffix}")
            fig.tight_layout(); fig.savefig(path); plt.close(fig)
            print(f"🖼 {path}")

    print("\n=== СВОДКА ===")
    for r in summary:
        print(" · ".join(f"{k}: {v}" for k, v in r.items()))


if __name__ == "__main__":
    main()
