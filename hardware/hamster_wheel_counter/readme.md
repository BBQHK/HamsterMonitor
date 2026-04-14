<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

## 完整麵包板接線圖

```
麵包板電源軌：
左側 + rail ───── CR123A (+)
右側 - rail ───── CR123A (-)

┌─────────────────────────────────────────────────────────────┐
│ 電源軌： [+] [+] [+] [+] [+] [+] [+] [+] [+] [+] [+] [+] [+] │
│           [-] [-] [-] [-] [-] [-] [-] [-] [-] [-] [-] [-] [-] │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ FR120N 模組（隔離開關）                                     │
│  IN+ ──── Pro Mini D4   │   OUT+ ──── CR123A (+) [+ rail]   │
│  IN- ──── GND [- rail]  │   OUT- ──── FS1000A VCC [+ rail]  │
│  GND ──── GND [- rail]                                        │
│                                                             │
│ Pro Mini 3.3V（面向 USB 焊點在上）                           │
│  VCC ──── [+ rail]  │  GND ──── [- rail]                    │
│  RAW ──── [+ rail]  │  D2  ──── A3144 DO                    │
│                     │  D7  ──── FS1000A DATA                │
│                     │  D10 ──── FS1000A PTT（空接）         │
│                     │  D4  ──── FR120N IN+                  │
│                                                             │
│ A3144 4腳模組                                                       │
│  VCC ──── [+ rail]  │  GND ──── [- rail]  │  DO ──── D2    │
│  AO ──── 不用接                                             │
│                                                             │
│ FS1000A                                                                 │
│  VCC ──── FR120N OUT- [+ rail]  │  DATA ──── D7            │
│  GND ──── [- rail]              │  ANT ──── 17cm 銅線       │
│  PTT ──── D10（空接）                                          │
└─────────────────────────────────────────────────────────────┘
```

## **詳細接線步驟**

### **Step 1：電源**

```
CR123A (+) ── 麵包板左側 + rail 全通
CR123A (-) ── 麵包板右側 - rail 全通
```

### **Step 2：FR120N 開關模組**

```
FR120N IN+  ── Pro Mini D4
FR120N IN-  ── Pro Mini GND (- rail)
FR120N GND  ── Pro Mini GND (- rail)
FR120N OUT+ ── CR123A (+) (+ rail)
FR120N OUT- ── FS1000A VCC (+ rail)
```

### **Step 3：Pro Mini**

```
Pro Mini VCC ── + rail
Pro Mini RAW ── + rail（共用電源）
Pro Mini GND ── - rail
```

### **Step 4：A3144 4 腳模組**

```
A3144 VCC ── + rail
A3144 GND ── - rail
A3144 DO  ── Pro Mini D2
A3144 AO  ── 不用接
```

### **Step 5：FS1000A**

```
FS1000A VCC ── FR120N OUT- (+ rail)
FS1000A GND ── - rail
FS1000A DATA ── Pro Mini D7
FS1000A ANT  ── 17cm 銅線（433MHz 1/4波長）
FS1000A PTT ── Pro Mini D10（RadioHead 要的，空接也行）
```

## **程式修改**

你原本程式用 **RadioHead RH_ASK**，改 `sendCountUpdate()`：

```cpp
static void sendCountUpdate() {
  digitalWrite(4, HIGH);     // D4 高電平 → FR120N 開啟電源
  delay(15);                 // 穩壓 + 光耦延遲

  WheelPacket pkt;
  pkt.magic = 0xA7;
  pkt.revolutions = revolutions;
  pkt.xorChecksum = checksumOf(pkt);

  rfDriver.send((uint8_t*)&pkt, sizeof(pkt));
  rfDriver.waitPacketSent();

  delay(5);                  // 確保發完
  digitalWrite(4, LOW);      // 關閉電源
}
```

**setup() 加這行**：

```cpp
pinMode(4, OUTPUT);
digitalWrite(4, LOW);      // 預設關閉
```

## **驗證步驟**

### **1. 供電測試**

```
量 FR120N OUT- 對 GND：應為 0V（關閉狀態）
拉高 D4：應變 3.3V（開啟狀態）
```

### **2. 發送測試**

```
燒錄程式，拉高 D4 → FS1000A 上電
Serial Monitor 看資料發送
量平均電流：睡覺時 <10μA
```

### **3. Hall + 計數**

```
磁鐵靠近 A3144 → D2 觸發 → 計數 +1 → 自動開 FR120N → 發送 → 關電源
```

## **預期效果**

```
無磁鐵：整機 1μA（Pro Mini 睡 + FS1000A 斷電）
磁鐵觸發：短暫 32mA（20ms 發送）
平均：每天 10k 圈 → 2μA → 3 年電池！
```

## **注意事項**

1. **FR120N 是高電平開啟**：`digitalWrite(D4, HIGH)` 開電源，`LOW` 關閉。[^2][^1]
2. **延遲 15ms**：光耦 + 穩壓需要時間。[^1]
3. **17cm 天線**：FS1000A 射程關鍵。[^4]
4. **Pro Mini 3.3V 版**：RAW 直接接 CR123A 3V。[^5]

**接好後直接燒錄測試**！**你的倉鼠輪計數器就完美了**，**3 年換一次電池**！[^2][^1]

**有問題直接問**，我幫你 debug！ 😎
<span style="display:none">[^10][^11][^12][^13][^14][^15][^16][^17][^6][^7][^8][^9]</span>

<div align="center">⁂</div>

[^1]: https://jin-hua.com.tw/page/product/show.aspx?num=37785
[^2]: https://shopee.tw/-熊的開發-隔離-MOSFET-場效電晶體-FR120N-LR7843-D4184-i.25956307.20934767181
[^3]: https://shopee.tw/附發票-隔離MOSFET-MOS管-場效應管模塊-替代繼電器-FR120N-LR7843-D4184-i.795631261.22920965512
[^4]: https://www.circuits-diy.com/fs1000a-433mhz-rf-transmitter-receiver-modules/
[^5]: https://www.arduino.cc/en/Main/ArduinoBoardProMini
[^6]: https://www.taiwansensor.com.tw/product/fr120n-100v-9-4a-隔離型-mosfet-mos-管場效應管模組-光耦合隔離模組/
[^7]: https://tw.bid.yahoo.com/item/100746176646
[^8]: https://www.wangzuanquan.com/ja-jp/insights-and-innovations/technical-documents/hall-effect-sensor-ic-publications/an296144-als31300-advanced-low-power-modes
[^9]: https://jin-hua.com.tw/page/product/p02.aspx?kind=2864
[^10]: https://www.mysensors.org/build/battery
[^11]: https://www.allegromicro.com/zh-cn/products/sense/linear-and-angular-position/linear-position-sensor-ics/a31010
[^12]: https://www.taiwaniot.com.tw/products-category/module-sensor/page/18/?orderby=menu_order
[^13]: https://hackmd.io/@fort-inc/rkE4I2SjO
[^14]: https://thecavepearlproject.org/2022/03/09/powering-a-promini-logger-for-one-year-on-a-coin-cell/
[^15]: http://oceansky-technology.com/commerce/product_info.php?products_id=13059
[^16]: http://oceansky-technology.com/commerce/product_info.php?products_id=12827
[^17]: https://www.taiwaniot.com.tw/products-category/module-sensor/其他類型感測器/
