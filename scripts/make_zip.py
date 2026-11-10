"""Genera el ZIP de entrega del proyecto y comprueba que esté completo.

Uso:  python scripts/make_zip.py            crea dist/proyecto_final_ia.zip
      python scripts/make_zip.py --dry-run  solo muestra qué se incluiría

Incluye el código fuente, el README, los modelos de inferencia, los reportes y el informe.
Excluye entornos virtuales, cachés, datos originales, el checkpoint de entrenamiento,
configuración local y cualquier archivo .env.
"""
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dist" / "proyecto_final_ia.zip"

INCLUDE_DIRS = ["app", "src", "tests", "scripts", "reports", "data/sample", "docs"]
INCLUDE_FILES = ["README.md", "requirements.txt", "requirements-dev.txt", "render.yaml",
                 ".python-version", ".gitignore", "pytest.ini", "wsgi.py",
                 "models/tokenizer.json", "models/emotion_bilstm_inference.keras",
                 "models/emotion_bilstm_weights.npz"]
EXCLUDE = re.compile(r"(__pycache__|\.pyc$|\.pytest_cache|(^|/)\.env|(^|/)~\$)")
REQUIRED = ["README.md", "docs/informe/Informe_Proyecto_Final_IA.pdf", "requirements.txt",
            "wsgi.py", "models/emotion_bilstm_weights.npz", "models/tokenizer.json"]
SECRET = re.compile(r"(api[_-]?key|secret[_-]?key|password|token)\s*[:=]\s*['\"][^'\"]{8,}",
                    re.IGNORECASE)
PLACEHOLDER = "Nombre completo del estudiante"


def collect():
    files = [ROOT / f for f in INCLUDE_FILES]
    for d in INCLUDE_DIRS:
        files += [p for p in (ROOT / d).rglob("*") if p.is_file()]
    rel = sorted({p.relative_to(ROOT).as_posix() for p in files if p.exists()})
    return [r for r in rel if not EXCLUDE.search(r)]


def checks(files):
    problems, warnings = [], []
    for req in REQUIRED:
        if req not in files:
            problems.append(f"Falta {req}")
    for f in files:
        if f.endswith((".py", ".yaml", ".txt", ".md", ".html", ".json")):
            text = (ROOT / f).read_text(encoding="utf-8", errors="ignore")
            if SECRET.search(text):
                problems.append(f"Posible secreto en {f}")
    report = ROOT / "docs/informe/Informe_Proyecto_Final_IA.docx"
    pdf = report.with_suffix(".pdf")
    if report.exists():
        xml = zipfile.ZipFile(report).read("word/document.xml").decode("utf-8", "ignore")
        if PLACEHOLDER in xml:
            warnings.append("La portada del informe aún tiene datos por completar (resaltados en amarillo)")
        if pdf.exists() and report.stat().st_mtime > pdf.stat().st_mtime + 60:
            warnings.append("El .docx es más reciente que el PDF: vuelve a exportar el PDF desde Word")
    return problems, warnings


def main():
    dry = "--dry-run" in sys.argv
    files = collect()
    total = sum((ROOT / f).stat().st_size for f in files)
    for f in files:
        print(f"  {f}")
    print(f"\n{len(files)} archivos, {total / 1e6:.1f} MB sin comprimir")

    problems, warnings = checks(files)
    for w in warnings:
        print(f"[AVISO] {w}")
    for p in problems:
        print(f"[ERROR] {p}")
    if problems:
        return 1
    if dry:
        print("Modo --dry-run: no se creó el ZIP.")
        return 0

    OUT.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(ROOT / f, f"proyecto_final_ia/{f}")
    print(f"ZIP creado: {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
