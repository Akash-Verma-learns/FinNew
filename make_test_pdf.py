from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

doc = SimpleDocTemplate("test_report.pdf", pagesize=A4)
styles = getSampleStyleSheet()
elements = []

elements.append(Paragraph("Apple Inc. — Equity Research Report FY2024", styles["Title"]))
elements.append(Spacer(1, 12))
elements.append(Paragraph("Investment Thesis: BUY | Target Price: $210 | Upside: 18%", styles["Normal"]))
elements.append(Spacer(1, 12))
elements.append(Paragraph("Apple reported record revenue of $391 billion in FY2024, gross margin of 46.2%, "
    "net income of $94 billion, and free cash flow of $108.8 billion. "
    "Cash and equivalents stood at $96.2 billion. EPS diluted was $6.11.", styles["Normal"]))
elements.append(Spacer(1, 12))

# Table 1 — Income Statement
elements.append(Paragraph("Income Statement Summary (USD billions)", styles["Heading2"]))
data1 = [
    ["Metric", "FY2022", "FY2023", "FY2024"],
    ["Revenue", "394.3", "383.3", "391.0"],
    ["Gross Profit", "170.8", "169.1", "180.7"],
    ["Gross Margin %", "43.3%", "44.1%", "46.2%"],
    ["Operating Income", "119.4", "114.3", "123.2"],
    ["Net Income", "99.8", "97.0", "94.0"],
    ["EPS (diluted)", "$6.15", "$6.13", "$6.11"],
]
t1 = Table(data1, colWidths=[150, 80, 80, 80])
t1.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.darkblue),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
    ("BACKGROUND", (0,1), (-1,-1), colors.lightgrey),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.lightgrey]),
]))
elements.append(t1)
elements.append(Spacer(1, 12))

# Table 2 — Segment Revenue
elements.append(Paragraph("Segment Revenue Breakdown (USD billions)", styles["Heading2"]))
data2 = [
    ["Segment", "FY2022", "FY2023", "FY2024", "YoY %"],
    ["iPhone", "205.5", "200.6", "210.0", "+4.7%"],
    ["Mac", "40.2", "29.4", "31.0", "+5.4%"],
    ["iPad", "29.3", "28.3", "26.9", "-4.9%"],
    ["Wearables", "41.2", "39.8", "37.0", "-7.0%"],
    ["Services", "78.1", "85.2", "86.1", "+1.1%"],
    ["Total", "394.3", "383.3", "391.0", "+2.0%"],
]
t2 = Table(data2, colWidths=[130, 75, 75, 75, 75])
t2.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.darkblue),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.lightgrey]),
]))
elements.append(t2)
elements.append(Spacer(1, 12))

# Table 3 — Valuation Multiples
elements.append(Paragraph("Valuation Multiples vs Peers", styles["Heading2"]))
data3 = [
    ["Company", "P/E (fwd)", "EV/EBITDA", "P/FCF", "Gross Margin"],
    ["Apple (AAPL)", "29x", "22x", "28x", "46.2%"],
    ["Microsoft (MSFT)", "31x", "24x", "32x", "69.4%"],
    ["Alphabet (GOOGL)", "21x", "16x", "22x", "56.9%"],
    ["Meta (META)", "24x", "15x", "20x", "81.0%"],
    ["Peer Median", "25x", "18x", "24x", "63.4%"],
]
t3 = Table(data3, colWidths=[140, 80, 80, 70, 80])
t3.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.darkblue),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("BACKGROUND", (0,1), (-1,1), colors.lightblue),
    ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
    ("ROWBACKGROUNDS", (0,2), (-1,-1), [colors.white, colors.lightgrey]),
]))
elements.append(t3)
elements.append(Spacer(1, 12))

# Table 4 — Balance Sheet
elements.append(Paragraph("Balance Sheet Key Metrics (USD billions)", styles["Heading2"]))
data4 = [
    ["Item", "FY2022", "FY2023", "FY2024"],
    ["Cash & Equivalents", "48.3", "61.6", "96.2"],
    ["Total Assets", "352.8", "352.6", "364.9"],
    ["Total Debt", "120.1", "111.1", "101.3"],
    ["Net Debt / (Cash)", "71.8", "49.5", "5.1"],
    ["Shareholders Equity", "50.7", "62.1", "78.4"],
    ["Debt-to-Equity", "2.37x", "1.79x", "1.29x"],
]
t4 = Table(data4, colWidths=[150, 80, 80, 80])
t4.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.darkblue),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.lightgrey]),
]))
elements.append(t4)

doc.build(elements)
print(f"PDF created: test_report.pdf")
