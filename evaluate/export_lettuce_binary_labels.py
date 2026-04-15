import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile
from xml.sax.saxutils import escape

from evaluate_lettuce_metrics import (
    LETTUCE_MODEL_NAME,
    calculate_metrics,
    evaluate_faithfulness,
    flatten_json_to_text,
)


def excel_column_name(index: int) -> str:
    name = ""
    while index > 0:
        index, rem = divmod(index - 1, 26)
        name = chr(65 + rem) + name
    return name


def write_simple_xlsx(path: Path, rows: list[dict[str, Any]], sheet_name: str) -> None:
    headers: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in headers:
                headers.append(key)

    shared_strings: list[str] = []
    shared_index: dict[str, int] = {}

    def shared_id(value: str) -> int:
        if value not in shared_index:
            shared_index[value] = len(shared_strings)
            shared_strings.append(value)
        return shared_index[value]

    def add_cell(cell_ref: str, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bool):
            return f'<c r="{cell_ref}" t="n"><v>{1 if value else 0}</v></c>'
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return f'<c r="{cell_ref}" t="n"><v>{value}</v></c>'
        value_str = str(value)
        if value_str == "":
            return ""
        sid = shared_id(value_str)
        return f'<c r="{cell_ref}" t="s"><v>{sid}</v></c>'

    row_xml: list[str] = []
    header_cells = []
    for col_idx, header in enumerate(headers, start=1):
        header_cells.append(add_cell(f"{excel_column_name(col_idx)}1", header))
    row_xml.append(f'<row r="1">{"".join(header_cells)}</row>')

    for row_idx, row in enumerate(rows, start=2):
        cells = []
        for col_idx, header in enumerate(headers, start=1):
            cells.append(add_cell(f"{excel_column_name(col_idx)}{row_idx}", row.get(header, "")))
        row_xml.append(f'<row r="{row_idx}">{"".join(cells)}</row>')

    shared_xml = "".join(f"<si><t>{escape(text)}</t></si>" for text in shared_strings)
    worksheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        "<sheetData>"
        + "".join(row_xml)
        + "</sheetData></worksheet>"
    )
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        "<sheets>"
        f'<sheet name="{escape(sheet_name)}" sheetId="1" r:id="rId1"/>'
        "</sheets></workbook>"
    )
    styles_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="2"><fill><patternFill patternType="none"/></fill>'
        '<fill><patternFill patternType="gray125"/></fill></fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>'
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        "</styleSheet>"
    )
    rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        "</Relationships>"
    )
    workbook_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/>'
        '<Relationship Id="rId2" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
        '<Relationship Id="rId3" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" '
        'Target="sharedStrings.xml"/>'
        "</Relationships>"
    )
    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        '<Override PartName="/xl/sharedStrings.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>'
        "</Types>"
    )
    shared_strings_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        f'count="{len(shared_strings)}" uniqueCount="{len(shared_strings)}">{shared_xml}</sst>'
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", rels_xml)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        zf.writestr("xl/worksheets/sheet1.xml", worksheet_xml)
        zf.writestr("xl/styles.xml", styles_xml)
        zf.writestr("xl/sharedStrings.xml", shared_strings_xml)


def binary_label(row: dict[str, Any]) -> int:
    counts = row.get("label_counts", {})
    return int(counts.get("unsupported", 0) > 0 or counts.get("contradicted", 0) > 0)


def sentence_summary(items: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"{item['label']} ({item['score']}): {item['sentence']}"
        for item in items
    )


def export_rows(rows: list[dict[str, Any]], answer_field: str) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        out.append(
            {
                "row_id": row["row_id"],
                "label": row["label"],
                "faithfulness_score": row.get("faithfulness_score"),
                "supported_count": row.get("label_counts", {}).get("supported", 0),
                "unsupported_count": row.get("label_counts", {}).get("unsupported", 0),
                "contradicted_count": row.get("label_counts", {}).get("contradicted", 0),
                "user_prompt": row.get("user_prompt", ""),
                "tool_call": row.get("tool_call", ""),
                "tool_response": row.get("tool_response", ""),
                answer_field: row.get(answer_field, ""),
                "faithfulness_eval_text": sentence_summary(row.get("faithfulness_eval", [])),
            }
        )
    return out


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        if not headers:
            return
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LettuceDetect and export row-level binary labels."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input dataset JSON.")
    parser.add_argument("--answer-field", required=True, help="Field to evaluate.")
    parser.add_argument("--output-prefix", type=Path, required=True, help="Output path prefix without extension.")
    parser.add_argument("--model", default=LETTUCE_MODEL_NAME, help="LettuceDetect model.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for the classifier.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from tqdm.auto import tqdm
    from transformers import pipeline as hf_pipeline

    rows = json.loads(args.input.read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading NLI model: {args.model} ...")
    nli_pipe = hf_pipeline(
        task="text-classification",
        model=args.model,
        device=0 if device == "cuda" else -1,
        truncation=True,
        max_length=512,
    )
    print("Model loaded.")

    evaluated = []
    for idx, row in enumerate(tqdm(rows, desc=f"Evaluating {args.answer_field}", unit="row")):
        new_row = deepcopy(row)
        context = flatten_json_to_text(row["tool_response"])
        eval_result = evaluate_faithfulness(
            nli_pipe=nli_pipe,
            question=row["user_prompt"],
            answer=row[args.answer_field],
            context=context,
            batch_size=args.batch_size,
        )
        new_row["row_id"] = idx
        new_row["faithfulness_eval"] = eval_result["sentences"]
        new_row["faithfulness_score"] = eval_result["faithfulness_score"]
        new_row["label_counts"] = eval_result["label_counts"]
        new_row["label"] = binary_label(new_row)
        evaluated.append(new_row)

    metrics = calculate_metrics(evaluated)
    exported = export_rows(evaluated, args.answer_field)

    write_json(args.output_prefix.with_suffix(".json"), evaluated)
    write_json(args.output_prefix.parent / f"{args.output_prefix.name}_summary.json", metrics)
    write_csv(args.output_prefix.with_suffix(".csv"), exported)
    write_simple_xlsx(args.output_prefix.with_suffix(".xlsx"), exported, "labels")

    print(f"Saved labeled JSON: {args.output_prefix.with_suffix('.json')}")
    print(f"Saved labeled CSV: {args.output_prefix.with_suffix('.csv')}")
    print(f"Saved labeled XLSX: {args.output_prefix.with_suffix('.xlsx')}")
    print(f"Saved summary JSON: {args.output_prefix.parent / f'{args.output_prefix.name}_summary.json'}")


if __name__ == "__main__":
    main()
