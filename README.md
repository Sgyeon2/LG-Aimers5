# LG-Aimers5
차량용 디스플레이 제조 공정 불량품 판별 모델 개발

## 개요 (Overview)
차량용 디스플레이 제조 공정에서 발생하는 **불량품(AbNormal) 판별**을 위해 분류 모델을 구축한 프로젝트입니다.  
하지만 아래와 같은 **주요 문제점**에 부딪혔고, 이를 해결하기 위한 다양한 방안을 적용했습니다.

---

## 문제 상황(Problem Statement) 및 해결 방법

### 1. Workorder별 불균형한 불량률
- **문제**  
  - 특정 Workorder에서 불량률이 유독 높게 나타남.
  - Workorder별 불량률을 미리 파악해 ‘높은 불량 위험군’에 대한 사전관리 필요성 대두.

- **해결**: **리스크 등급화(Risk Category)**
  1. **Workorder_Dam 별 불량 비율 산출**  
     - `pd.crosstab`으로 Workorder와 target(`Normal`, `AbNormal`) 간 빈도수를 계산.  
     - 불량 비율(`Abnormal_Percentage`) = (해당 Workorder의 AbNormal 건수 / 전체 건수) × 100
  2. **불량 비율 기반 ‘리스크 범주’ 설정**  
     - 예: 75% 이상 → High Risk(2), 25% 이상 75% 미만 → Normal Risk(1), 25% 미만 → Low Risk(0)  
     - `Risk_Category` 컬럼 추가하여 모델에 함께 입력
  3. **결과**  
     - Workorder별 위험도를 반영할 수 있어 모델 성능이 개선되고, 위험군에 대한 사전 대응이 용이

---

### 2. 클래스 불균형(9:1)으로 인한 모델 편향
- **문제**  
  - 데이터셋에서 Normal(정상품)이 9, AbNormal(불량품)이 1의 비율로 불균형함.  
  - 모델이 단순히 Normal 위주로 학습되어, 불량품 예측 정확도가 떨어질 위험 발생.

- **해결**  
  1. **Stratify 옵션 사용**  
     - `train_test_split` 시 `stratify=y`를 적용하거나, K-Fold 교차검증에서 **Stratified K-Fold**를 사용해  
       훈련·검증·테스트 세트에 동일한 클래스 비율을 유지.
  2. **XGBoost의 `scale_pos_weight`**  
     - 불량 데이터의 클래스 비중을 실제 비율 대비 높여서(**scale_pos_weight=10** 등)  
       모델이 불량 클래스를 더 적극적으로 학습하도록 유도.
  3. **Threshold 조정**  
     - 분류 모델이 예측한 확률(`predict_proba`)을 기준으로, 0.5 대신 다양한 임계값을 시험해 봄.  
     - 특히 **불량(AbNormal)에 대한 F1 스코어** 최적화 시점을 찾고, 그 임계값(`best_threshold`)을 적용하여 적절한 분류 성능 확보.

---

### 3. 과적합(Overfitting) 발생 가능성
- **문제**  
  - 트리 기반 모델(XGBoost, CatBoost, LightGBM, RandomForest)을 함께 쓰는 상황에서,  
    고차원·복잡한 모델이 훈련 데이터에만 지나치게 맞춰져 과적합될 우려가 큼.

- **해결**  
  1. **Voting Classifier(Soft Voting)**  
     - 단일 모델만 쓰지 않고, XGBoost / RandomForest / CatBoost / LightGBM을 앙상블(Soft Voting)  
     - 서로 다른 알고리즘 결합으로 **모델 편향 감소**, 과적합을 줄임.
  2. **하이퍼파라미터 제한**  
     - 각 모델에서 트리 깊이(`max_depth`), 최소 샘플 분할(`min_samples_split`) 등 제한  
     - 학습률(`learning_rate`)을 작게 두고, 규제 항목(`gamma`, `l2_leaf_reg` 등)을 적용해 모델 복잡도 억제.
  3. **Cross Validation(교차검증)**  
     - 모델 학습 시 단순히 Train/Test로 나누는 것 외에, **교차검증**으로 다양한 데이터 샘플에 대해 성능 확인  
     - 평균 성능과 표준편차를 함께 살펴, 특정 Fold에만 성능이 치우치지 않는지 점검.
      
---

## 결론 
  - Workorder별 리스크 등급화(Risk_Category)로 공정 특성을 반영해 예측력을 개선  
  - 클래스 불균형(9:1) 문제 해결을 위해 Stratify, XGBoost 파라미터 조정, Threshold 튜닝 등을 병행  
  - 앙상블 모델(Voting Classifier)로 과적합을 완화하고 안정적 성능 확보
