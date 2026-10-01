# Core Playbook Dow & PA (bản rút gọn — đủ gửi LLM)

Nguồn: Edwards & Magee + Al Brooks + Bob Volman. Mục tiêu: BTCUSDT Futures 4H hướng / 1H entry.

## 1. Regime & Dow (bắt buộc trước khi vào lệnh)

**4 trạng thái:** Spike/Strong Trend | Channel | Trading Range | Breakout/Squeeze.

**Dow:**
- Uptrend: HH + HL → chỉ ưu tiên BUY
- Downtrend: LH + LL → chỉ ưu tiên SELL
- Sideway: đỉnh/đáy chồng → HOLD trừ trap/setup cực nét tại biên
- Chưa phá HL→LL hoặc LH→HH → xu hướng cũ còn hiệu lực

**Key magnets:** swing H/L, S/R, EMA20/25, số tròn. Chỉ vào lệnh PA **tại** key level / EMA, không giữa khoảng trống.

**Micro channel (≥5 nến không phá cực trước):** chỉ trade thuận hướng.

## 2. Setup PA ưu tiên (IF → THEN)

Chỉ BUY theo uptrend 4H / SELL theo downtrend 4H.

| Setup | Điều kiện | Entry | SL | TP |
|-------|-----------|-------|----|----|
| Pinbar | Râu ≥50% range, close 1/3 đầu kia, tại key level | Stop ngoài extreme pin | Ngoài chóp râu | S/R gần nhất, R:R≥1.5 |
| Engulfing/OB | Bao trùm H-L nến trước, close sát biên, tại level | Stop ngoài extreme OB | Ngoài đáy/đỉnh OB | R:R≥1.5 |
| Inside ii/iii | Nén sát cản/EMA | Stop phá cụm | Phía đối diện cụm | 1.5–2× range cụm |
| 2BR | 2 trend bar ngược tại cản | Stop ngoài nến 2 | Ngoài cực 2 nến | Swing gần nhất |
| MDB/MDT | 2 low/high gần bằng tại EMA/S/R | Stop ngoài nến 2 | Ngoài đáy/đỉnh kép | R:R≥1.5 |
| Marubozu | Body >80%, breakout | Stop ngoài extreme / market close | Mid hoặc đáy/đỉnh bar | Measured move |
| High2 / Low2 | Pullback ABC trong trend + EMA | Stop ngoài signal H2/L2 | Ngoài đáy H2 / đỉnh L2 | Swing cũ |
| Wedge H3/L3 | 3 push kiệt tại trendline/EMA | Stop ngoài signal push3 | Ngoài cực push3 | Đầu wedge |
| Volman PB | Buildup ≥4 sát cản + EMA25 | Stop ngoài cản | Ngoài cụm buildup | R:R≥1.5 |
| PBP | Break rồi ceiling-test giữ | Stop ngoài nến test | Ngoài nến test | Mở rộng break |
| PBC | Break nhỏ + bounce EMA25 | Stop ngoài trigger | Ngoài điểm EMA | Biên range lớn |
| PBR | Angular PB về EMA25 + reject | Stop ngoài signal | Ngoài râu signal | Swing gần |
| Trap | Sweep S/R rồi đóng ngược vào range | Stop ngoài trap bar | Ngoài sweep extreme | Biên đối diện |
| H&S / DTop-Bot | Pattern + đóng thủng neckline | Break/retest neckline | Vai phải / cực pattern | Measured move |

## 3. Fail / Scratch (cắt sớm — không chờ SL mù)

- Pin/Engulf: nến sau đóng ngược >50% thân setup → scratch
- Inside break rồi đóng phía đối diện cụm → false break → thoát
- 2BR bị nến 3 nuốt lại → fail
- MDB/MDT xuyên mức kép → cắt
- Marubozu: nến sau đóng qua mid (50%) ngược hướng → fail momentum
- H2/L2 bị thủng rồi kéo dài thành H3/L3 → thoát nếu signal bị vi phạm
- PB/PBP/PBC/PBR: chựng trong buildup / xuyên lại EMA25 / ceiling fail → scratch
- Trap chạy thất bại (phá lại sweep) → thoát
- H&S: sau break neckline mà đóng vượt vai phải → thoát hết

## 4. Risk & Skip (hard)

- SL **kỹ thuật** (ngoài signal/buildup/swing) — cấm nới SL khi đang lỗ
- R:R < 1.5 → HOLD/SKIP
- Size: risk% / |entry−SL| (bot paper: margin cố định 100u)
- Breakeven chỉ sau +1R **và** breakout-test OK; trail theo basing HL/LH
- **SKIP:** break không buildup; ngược powerbar cluster; ± tin HIGH impact; range hẹp vô định → HOLD

## 5. Quy trình quyết định bot (1 lệnh)

1. Regime 4H (U/D/S/News) → bias
2. Key level Dow trên 4H+1H
3. PA setup bảng mục 2 tại level, thuận bias
4. News HIGH → nâng điểm / HOLD
5. Score ≥ min_score → BUY/SELL kèm entry/SL/TP; dưới min → HOLD
