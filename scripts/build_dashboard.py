"""Inject results/report.json + results/sim_data.json into dashboard/template.html."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def build() -> Path:
    tpl = (ROOT / "dashboard" / "template.html").read_text(encoding="utf-8")
    rep = (ROOT / "results" / "report.json").read_text(encoding="utf-8").replace("</", "<\\/")
    sim = (ROOT / "results" / "sim_data.json").read_text(encoding="utf-8").replace("</", "<\\/")
    out = ROOT / "dashboard" / "edgelab_dashboard.html"
    out.write_text(tpl.replace("/*__REPORT__*/", rep).replace("/*__SIM__*/", sim), encoding="utf-8")
    return out


if __name__ == "__main__":
    print(build())
