"""Optional host-side reporting, outside the selector's default hot path."""
import statistics


def summarize(records):
    n = len(records)
    active = [r for r in records if not r.get("bypassed", False)]
    overflow = sum(r["union_size"] > r["budget"] for r in active)
    shortfall = sum(r["union_size"] < r["budget"] for r in active)
    result = dict(records=n, selection_records=len(active),
                  samples=len({(r.get("task"), r.get("sample_index")) for r in records}),
                  bypassed=n-len(active), overflow=overflow, shortfall=shortfall,
                  overflow_rate=overflow/len(active) if active else 0.,
                  shortfall_rate=shortfall/len(active) if active else 0.,
                  strict_final_violations=sum(r.get("budget_applied", True) and r["strict_budget"] and
                      r["selected_size"] > r["budget"] for r in records))
    for key in ("union_size", "selected_size", "candidate_length", "prefix_length"):
        values = sorted(r[key] for r in records)
        if not values:
            result[key] = None
            continue
        pos = (len(values)-1)*.95
        i = int(pos)
        p95 = values[i] + (values[min(i+1,len(values)-1)]-values[i])*(pos-i)
        result[key] = dict(min=values[0], mean=statistics.mean(values),
                           median=statistics.median(values), p95=p95, max=values[-1])
    return result
