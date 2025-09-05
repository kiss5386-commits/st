#!/usr/bin/env python3
"""
ROI% 기준 손절/트레일링 계산 테스트

ROI% → Price% 변환 공식: price_pct = roi_pct / leverage
"""

def test_roi_calculations():
    """ROI% 기준 계산 테스트 시나리오"""
    
    test_cases = [
        {
            "name": "10배 레버리지, ROI 15% 손절",
            "leverage": 10,
            "roi_pct": 15,
            "entry_price": 100.0,
            "side": "long"
        },
        {
            "name": "5배 레버리지, ROI 10% 트리거",
            "leverage": 5,
            "roi_pct": 10,
            "entry_price": 50000.0,
            "side": "long"
        },
        {
            "name": "20배 레버리지, ROI 5% 콜백",
            "leverage": 20,
            "roi_pct": 5,
            "entry_price": 2.5,
            "side": "short"
        }
    ]
    
    print("=== ROI% → Price% 변환 테스트 ===\n")
    
    for case in test_cases:
        leverage = case["leverage"]
        roi_pct = case["roi_pct"]
        entry_price = case["entry_price"]
        side = case["side"]
        
        # ROI% → Price% 변환
        price_pct = roi_pct / leverage
        
        # 실제 가격 계산
        if side == "long":
            target_price = entry_price * (1 - price_pct / 100.0) if "손절" in case["name"] else \
                          entry_price * (1 + price_pct / 100.0)
        else:  # short
            target_price = entry_price * (1 + price_pct / 100.0) if "손절" in case["name"] else \
                          entry_price * (1 - price_pct / 100.0)
        
        # ROI 검증 계산
        price_change_pct = abs(target_price - entry_price) / entry_price * 100
        actual_roi = price_change_pct * leverage
        
        print(f"📊 {case['name']}")
        print(f"   레버리지: {leverage}배")
        print(f"   목표 ROI: {roi_pct}%")
        print(f"   → Price%: {price_pct:.2f}%")
        print(f"   진입가: ${entry_price}")
        print(f"   목표가: ${target_price:.4f}")
        print(f"   가격변동: {price_change_pct:.2f}%")
        print(f"   실제ROI: {actual_roi:.1f}%")
        print(f"   ✅ ROI 일치: {'예' if abs(actual_roi - roi_pct) < 0.1 else '❌ 아니오'}")
        print()

if __name__ == "__main__":
    test_roi_calculations()