# predict-the-introverts-extroverts-analysis
내향인과 외향인 예측 분석 프로젝트(kaggle playground, 2025)

## 1. 프로젝트 개요 (Introduction)

* **분석 배경 및 목적:**
    * 이 프로젝트는 Kaggle의 'Playground Series - S5E7' 참여를 위해 진행되었습니다.
    * 주어진 대회 데이터를 바탕으로, 응답자의 성격 유형(내향형/외향형)을 예측하는 머신러닝 모델을 개발하고 성능을 최적화하는 것을 목표로 합니다.
* **분석 기간:** 2025-06-14 ~ 2025-06-25
* **사용 데이터:** [Kaggle Playground Series - S5E7 Dataset](https://www.kaggle.com/competitions/playground-series-s5e7/data), [Extrovert vs. Introvert Behavior Data](https://www.kaggle.com/datasets/rakeshkapilavai/extrovert-vs-introvert-behavior-data/data)


## 2. 주요 분석 내용 (Key Features)

### 가. 데이터 전처리 (Data Preprocessing)

* **타겟 변수 인코딩:** 예측 대상인 `Personality` 컬럼을 머신러닝 모델이 학습할 수 있도록 'Introvert'는 1, 'Extrovert'는 0으로 변환했습니다.
* **결측치 처리:** 데이터의 특성을 고려하여 결측치 처리 전략을 다르게 적용했습니다.
    * `Friends_circle_size`, `Post_frequency`와 같이 연속적인 분포를 가질 가능성이 있는 컬럼은 **평균값(mean)**으로 대체했습니다.
    * `Going_outside`, `Social_event_attendance` 등 개인의 성향을 나타내는 컬럼은 극단치의 영향을 덜 받는 **중앙값(median)**으로 대체했습니다.
* **범주형 변수 처리:**
    * 'Yes'/'No' 형태의 `Stage_fear`, `Drained_after_socializing` 컬럼을 숫자 1/0으로 변환했습니다.
    * 변환 후에도 남아있는 결측치는 가장 빈도가 높은 값인 **최빈값(mode)**으로 채워 데이터의 일관성을 유지했습니다.
* **데이터 확장 (Data Augmentation):**
    * 모델의 학습 데이터 양을 늘리고 일반화 성능을 높이기 위해, 주어진 대회용 훈련 데이터 외에 **원본 데이터셋을 추가로 병합**하여 전체 훈련 데이터를 구성했습니다.


### 나. 탐색적 데이터 분석 (EDA)
* 각 변수(Feature)가 예측 대상(Target)인 성격 유형과 어떤 관계가 있는지 시각화를 통해 보여줍니다.
* **가장 중요하고 흥미로운 발견을 한두 개 정도** 뽑아서 그래프 이미지와 함께 설명합니다.
    * (예시) "분석 결과, '친구들과의 약속' 관련 질문에서 특정 답변을 한 그룹이 외향형일 확률이 월등히 높게 나타났습니다."
    * 
### 다. 모델링 (Modeling)
단일 모델의 한계를 극복하고 예측 안정성과 정확도를 높이기 위해 **앙상블(Ensemble)** 기법을 핵심 전략으로 사용했습니다.

* **특성 스케일링 (Feature Scaling):**
    * 모델 훈련 전, `StandardScaler`를 이용해 모든 특성(feature)의 스케일을 통일했습니다. 이를 통해 특정 변수가 모델에 과도한 영향을 미치는 것을 방지하고 안정적인 학습을 유도했습니다.
* **앙상블 모델 구축:**
    * 성능이 검증된 3개의 그래디언트 부스팅 모델(`CatBoost`, `XGBoost`, `LightGBM`)을 기본 모델로 선정했습니다.
    * `VotingClassifier`를 사용하여 세 모델의 예측 확률값을 종합적으로 고려하는 **소프트 보팅(Soft Voting)** 방식을 채택하여 최종 예측의 신뢰도를 높였습니다.

* 모델 성능을 어떻게 평가했는지 지표를 함께 보여줍니다. (예: Accuracy, F1-Score, ROC-AUC 등)
    * **Kaggle Public Score / Private Score:** XXX점

## 3. 결론 및 인사이트 (Conclusion & Insights)

* **프로젝트 요약:**
    * 원본 데이터를 포함한 확장된 데이터셋을 구성하고, 3개의 강력한 부스팅 모델(CatBoost, XGBoost, LightGBM)을 소프트 보팅 방식으로 앙상블하여 내/외향인 예측 모델을 구축했습니다. 
* **인사이트:**
    * 단일 모델보다 여러 모델의 예측을 결합하는 앙상블 기법이 더 안정적이고 높은 성능을 보임을 확인할 수 있었습니다.
    * **데이터의 양과 질이 모델 성능에 미치는 중요성**을 체감했으며, 외부 데이터를 적극적으로 활용하는 전략이 캐글과 같은 예측 대회에서 유효함을 확인했습니다.
* **Kaggle 최종 점수:**
    * Public Score: **0.974089**
    * Private Score: **0.967611**

* **인사이트:** 이 프로젝트를 통해 무엇을 배울 수 있었는지, 어떤 점이 아쉬웠는지 등을 자유롭게 작성합니다.
    * (예시) "X 변수가 예측에 가장 중요한 영향을 미치는 것을 확인했으며, Y 피처 엔지니어링을 통해 성능을 0.X% 향상시킬 수 있었습니다. 다만, Z 모델의 하이퍼파라미터 튜닝에 더 많은 시간을 투자했다면 순위를 더 높일 수 있었을 것입니다."


## 4. 사용 기술 (Tech Stack)

* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, CatBoost, XGBoost, LightGBM, Matplotlib, Seaborn
* **Tool:** databricks(Notebook)

---
