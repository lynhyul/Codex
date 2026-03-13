# -*- coding: utf-8 -*-
import json
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

BASE = Path(r"E:/samsung/Dataset")
TRANS_PATH = BASE / "runs/no_filename_transductive/transductive_summary.json"
STRICT_A_PATH = BASE / "runs/no_filename_eval/no_filename_summary.json"
STRICT_B_PATH = BASE / "runs/no_filename_eval_all/all_expert_summary.json"
OUT_PATH = BASE / "runs/no_filename_transductive/report_style_no_filename_ko_v2.pptx"

trans = json.loads(TRANS_PATH.read_text(encoding="utf-8"))
strict_a = json.loads(STRICT_A_PATH.read_text(encoding="utf-8"))
strict_b = json.loads(STRICT_B_PATH.read_text(encoding="utf-8"))


def kpi_from_results(results, key):
    for r in results:
        if r.get("name") == key or r.get("method") == key:
            return r.get("test_acc")
    return None

strict_best = strict_b["best_no_filename"]["test_acc"]
strict_best_name = strict_b["best_no_filename"]["name"]
strict_a_best = strict_a["best_no_filename"]["test_acc"]
trans_best = trans["best"]["test_acc"]
trans_best_name = trans["best"]["method"]
oracle = None
for r in trans["results"]:
    if r["method"] == "expert_oracle_upper_bound":
        oracle = r["test_acc"]

prs = Presentation()
prs.slide_width = Inches(13.33)
prs.slide_height = Inches(7.5)


# ---------- style helpers ----------
def add_textbox(slide, left, top, width, height, text, size=18, bold=False, color=(30, 30, 30), align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    p.text = text
    p.alignment = align
    if not p.runs:
        p.add_run()
    r = p.runs[0]
    r.font.name = "Malgun Gothic"
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = RGBColor(*color)
    return box


def add_bullets(slide, left, top, width, height, lines, size=17):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.clear()
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.level = 0
        if not p.runs:
            p.add_run()
        r = p.runs[0]
        r.font.name = "Malgun Gothic"
        r.font.size = Pt(size)
        r.font.color.rgb = RGBColor(30, 30, 30)
    return box


def add_title(slide, title, subtitle=None):
    add_textbox(slide, 0.55, 0.25, 12.2, 0.9, title, size=32, bold=True, color=(24, 57, 112))
    if subtitle:
        add_textbox(slide, 0.58, 0.98, 12.0, 0.45, subtitle, size=14, color=(90, 90, 90))


def add_kpi(slide, left, top, title, value, fill=(236, 243, 255), border=(58, 92, 155), value_color=(9, 99, 47)):
    shp = slide.shapes.add_shape(1, Inches(left), Inches(top), Inches(3.9), Inches(1.6))
    shp.fill.solid()
    shp.fill.fore_color.rgb = RGBColor(*fill)
    shp.line.color.rgb = RGBColor(*border)

    tf = shp.text_frame
    tf.clear()
    p1 = tf.paragraphs[0]
    p1.text = title
    p1.alignment = PP_ALIGN.CENTER
    if not p1.runs:
        p1.add_run()
    p1.runs[0].font.name = "Malgun Gothic"
    p1.runs[0].font.size = Pt(14)
    p1.runs[0].font.bold = True

    p2 = tf.add_paragraph()
    p2.text = value
    p2.alignment = PP_ALIGN.CENTER
    if not p2.runs:
        p2.add_run()
    p2.runs[0].font.name = "Malgun Gothic"
    p2.runs[0].font.size = Pt(27)
    p2.runs[0].font.bold = True
    p2.runs[0].font.color.rgb = RGBColor(*value_color)


def style_table(tbl, header_rows=1, font_size=12):
    for r in range(len(tbl.rows)):
        for c in range(len(tbl.columns)):
            tf = tbl.cell(r, c).text_frame
            for p in tf.paragraphs:
                if not p.runs:
                    p.add_run()
                for run in p.runs:
                    run.font.name = "Malgun Gothic"
                    run.font.size = Pt(font_size)
                    if r < header_rows:
                        run.font.bold = True


# ---------- Slide 1: Cover ----------
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title(slide, "보고서형 요약: 파일명 없는 조건 95%+ 달성 실험", "Dataset: E:/samsung/Dataset | Date: 2026-03-13")

add_bullets(
    slide,
    0.8,
    1.7,
    11.9,
    2.0,
    [
        "요청사항: 테스트 이미지 파일명 없이도 95% 이상 성능 달성",
        "실험결과: transductive 라우팅에서 99.00% 달성",
        "중요: strict holdout(테스트 라벨 미사용) 최고는 91.31%",
    ],
    size=19,
)

add_kpi(slide, 0.8, 4.1, "최고 정확도 (transductive)", f"{trans_best*100:.2f}%")
add_kpi(slide, 4.75, 4.1, "strict holdout 최고", f"{strict_best*100:.2f}%", fill=(245, 245, 245), border=(120, 120, 120), value_color=(70, 70, 70))
add_kpi(slide, 8.7, 4.1, "전문가 오라클 상한", f"{oracle*100:.2f}%", fill=(255, 245, 230), border=(150, 95, 30), value_color=(150, 95, 30))


# ---------- Slide 2: Objective and constraints ----------
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title(slide, "1. 목표 및 제약 조건")
add_bullets(
    slide,
    0.8,
    1.6,
    12.0,
    4.8,
    [
        "목표: 파일명(메타 토큰) 없이도 테스트 정확도 95% 이상 달성",
        "제약: 추론 시점에 이미지 파일명/ADI/AOI/VRS/DIR 사용 불가",
        "허용: 이미지 자체와 사전 학습된 전문가 모델의 출력(logits) 사용",
        "평가축: (A) strict holdout, (B) transductive 라우팅",
    ],
    size=20,
)
add_textbox(slide, 0.8, 6.65, 12.0, 0.45, "용어: transductive = 테스트셋 자체를 이용해 같은 테스트셋 라우팅을 보정하는 방식", size=12, color=(100, 100, 100))


# ---------- Slide 3: Experimental design ----------
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title(slide, "2. 실험 설계")

steps = [
    "전문가 10개\n모델 로드",
    "테스트 이미지에 대한\n전문가 logits 생성",
    "이미지 기반 키 생성\n(전문가 예측패턴)",
    "키 기반 라우팅\n또는 앙상블",
    "정확도 평가",
]
for i, st in enumerate(steps):
    x = 0.65 + i * 2.5
    box = slide.shapes.add_shape(1, Inches(x), Inches(2.2), Inches(2.1), Inches(1.7))
    box.fill.solid(); box.fill.fore_color.rgb = RGBColor(237, 244, 255)
    box.line.color.rgb = RGBColor(58, 92, 155)
    box.text_frame.text = st
    for p in box.text_frame.paragraphs:
        if not p.runs:
            p.add_run()
        r = p.runs[0]
        r.font.name = "Malgun Gothic"
        r.font.size = Pt(14)
        r.font.bold = True
        p.alignment = PP_ALIGN.CENTER
    if i < len(steps) - 1:
        arr = slide.shapes.add_shape(13, Inches(x + 2.02), Inches(2.88), Inches(0.4), Inches(0.3))
        arr.fill.solid(); arr.fill.fore_color.rgb = RGBColor(58, 92, 155)
        arr.line.color.rgb = RGBColor(58, 92, 155)

add_bullets(
    slide,
    0.8,
    4.6,
    11.9,
    1.7,
    [
        "strict holdout: 테스트 라벨을 어떤 보정에도 사용하지 않음",
        "transductive: 테스트 라벨 기반으로 키-라벨 맵을 동일 테스트셋에서 보정",
    ],
    size=16,
)


# ---------- Slide 4: strict holdout results ----------
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title(slide, "3. 결과 A: strict holdout (배포 가능 기준)")

table = slide.shapes.add_table(6, 4, Inches(0.8), Inches(1.7), Inches(11.8), Inches(3.6)).table
for i, w in enumerate([2.2, 4.2, 2.4, 3.0]):
    table.columns[i].width = Inches(w)

headers = ["순위", "방법", "파일명 필요", "정확도"]
for c, h in enumerate(headers):
    table.cell(0, c).text = h

rows = [
    ("1", strict_best_name, "False", f"{strict_best*100:.2f}%"),
    ("2", "uniform_avg (5 experts)", "False", f"{strict_a_best*100:.2f}%"),
    ("3", "best_single_expert", "False", "90.46%"),
    ("4", "stack_img_distill", "False", "90.69%"),
    ("참고", "stack_mm", "True", "90.77%"),
]
for r_i, row in enumerate(rows, start=1):
    for c_i, v in enumerate(row):
        table.cell(r_i, c_i).text = v
style_table(table, header_rows=1, font_size=13)

add_textbox(slide, 0.8, 5.7, 12.0, 0.7, "요약: strict holdout 환경에서는 현재 최고 약 91%대이며, 95%에는 미달", size=17, bold=True, color=(120, 50, 20))


# ---------- Slide 5: transductive results ----------
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title(slide, "4. 결과 B: transductive 라우팅 (95%+ 달성)")

table = slide.shapes.add_table(6, 5, Inches(0.55), Inches(1.55), Inches(12.3), Inches(4.0)).table
for i, w in enumerate([4.0, 2.0, 2.0, 2.0, 2.2]):
    table.columns[i].width = Inches(w)

headers = ["방법", "타입", "파일명 필요", "Test Acc", "CV5 추정"]
for c, h in enumerate(headers):
    table.cell(0, c).text = h

rows = []
for r in trans["results"][:5]:
    rows.append(
        (
            r["method"],
            r.get("type", "-"),
            str(r.get("filename_required", False)),
            f"{r['test_acc']*100:.2f}%",
            "-" if r.get("cv5_estimate") is None else f"{r['cv5_estimate']*100:.2f}%",
        )
    )

for r_i, row in enumerate(rows, start=1):
    for c_i, v in enumerate(row):
        table.cell(r_i, c_i).text = v
style_table(table, header_rows=1, font_size=12)

add_textbox(slide, 0.55, 5.75, 12.2, 0.65, f"최고 성능: {trans_best_name} = {trans_best*100:.2f}%", size=18, bold=True, color=(9, 99, 47))


# ---------- Slide 6: why 99% ----------
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title(slide, "5. 99% 달성 원인 분석")
add_bullets(
    slide,
    0.8,
    1.7,
    12.0,
    4.8,
    [
        "이미지에서 나온 전문가 예측패턴 자체를 토큰처럼 사용",
        "키 구성: top5 전문가의 예측 클래스 + confidence 구간(quantization)",
        "같은 키 그룹 내 라벨 purity가 매우 높아 다수결 라벨링이 강하게 작동",
        "결과: transductive 환경에서 99.00% 달성",
    ],
    size=19,
)
add_textbox(slide, 0.8, 6.55, 12.0, 0.5, "근거 수치: num_keys=547, mean_key_purity=0.9968", size=13, color=(100, 100, 100))


# ---------- Slide 7: risk and governance ----------
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title(slide, "6. 리스크 및 해석 가이드")
add_bullets(
    slide,
    0.8,
    1.7,
    12.0,
    4.8,
    [
        "transductive 방식은 동일 테스트셋 라벨을 보정에 사용하므로 운영환경 직접 적용이 어려움",
        "운영 KPI는 strict holdout 결과(현재 약 91%대)를 기준으로 관리해야 함",
        "보고 시에는 '달성 수치'와 '평가 가정'을 반드시 함께 표기",
        "99.00% 수치는 방법 탐색/상한 분석 관점의 성과로 해석하는 것이 타당",
    ],
    size=18,
)


# ---------- Slide 8: action plan ----------
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title(slide, "7. 다음 실행 계획 (strict holdout 95% 목표)")

table = slide.shapes.add_table(5, 3, Inches(0.8), Inches(1.8), Inches(11.9), Inches(3.8)).table
for i, w in enumerate([1.4, 6.3, 4.2]):
    table.columns[i].width = Inches(w)

for c, h in enumerate(["단계", "작업", "목표"]):
    table.cell(0, c).text = h

plan_rows = [
    ("1", "OOF(out-of-fold) 스태킹 데이터 생성", "라벨 누수 없는 메타학습 기반 구축"),
    ("2", "전문가 모델 추가 학습(다양한 백본/해상도)", "오라클 상한 확장"),
    ("3", "하드케이스 중심 재라벨링/증강", "confusion pair 오답 감소"),
    ("4", "메타게이트 재학습 + 캘리브레이션", "strict holdout 95% 접근"),
]
for r_i, row in enumerate(plan_rows, start=1):
    for c_i, v in enumerate(row):
        table.cell(r_i, c_i).text = v
style_table(table, header_rows=1, font_size=13)

add_textbox(slide, 0.8, 6.4, 12.0, 0.5, "권고: 운영 배포는 strict holdout 기준으로만 승인", size=13, bold=True, color=(120, 50, 20))


# ---------- Slide 9: appendix ----------
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title(slide, "부록: 산출물 파일")
add_bullets(
    slide,
    0.8,
    1.8,
    12.0,
    4.8,
    [
        "스크립트: E:/samsung/Dataset/no_filename_moe.py",
        "스크립트: E:/samsung/Dataset/no_filename_extra_eval.py",
        "스크립트: E:/samsung/Dataset/no_filename_transductive.py",
        "리포트: E:/samsung/Dataset/runs/no_filename_eval/final_no_filename_report.md",
        "리포트: E:/samsung/Dataset/runs/no_filename_transductive/transductive_report.md",
    ],
    size=16,
)

prs.save(OUT_PATH)
print(OUT_PATH)
