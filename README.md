# 내향인과 외향인 예측 분석(2025)

## 1. 프로젝트 개요

* **분석 배경 및 목적:**
    * 이 프로젝트는 Kaggle의 'Playground Series - S5E7' 참여를 위해 진행됨
    * 주어진 대회 데이터를 바탕으로, 응답자의 성격 유형(내향형/외향형)을 예측하는 머신러닝 모델을 개발 및 성능 최적화를 목표함
* **분석 기간:** 2025-06-14 ~ 2025-06-25
* **사용 데이터:** [Kaggle Playground Series - S5E7 Dataset](https://www.kaggle.com/competitions/playground-series-s5e7/data), [Extrovert vs. Introvert Behavior Data](https://www.kaggle.com/datasets/rakeshkapilavai/extrovert-vs-introvert-behavior-data/data)


   | 컬럼명                             | 한글 해석      | 설명                                            |
   | ------------------------------- | ---------- | --------------------------------------------- |
   | **id**                          | 아이디        | 각 데이터를 구분하는 고유 번호 (학생번호, 설문번호 등)              |
   | **Time\_spent\_Alone**          | 혼자 보낸 시간   | 혼자 있는 데 쓴 시간 (하루/일주일 단위 등, 단위는 데이터에 따라 다름)    |
   | **Stage\_fear**                 | 무대 공포      | 무대 공포(사람들 앞에서 발표하거나 공연할 때 두려움이 있는지 여부, 혹은 점수) |
   | **Social\_event\_attendance**   | 사교 행사 참석   | 사교적 행사(모임, 파티, 동호회 등) 참석 빈도 또는 횟수             |
   | **Going\_outside**              | 외출 빈도      | 집 밖으로 나가는 빈도, 얼마나 자주 외출하는지                    |
   | **Drained\_after\_socializing** | 사회활동 후 피로감 | 사회활동(사람들과 어울림) 후에 피곤함을 느끼는지 여부 또는 정도          |
   | **Friends\_circle\_size**       | 친구 모임 크기   | 가까운 친구의 수, 친구 집단(모임)의 규모                      |
   | **Post\_frequency**             | 게시물 업로드 빈도 | SNS 등에서 게시물(글, 사진 등)을 얼마나 자주 올리는지             |


## 2. 주요 분석 내용

### 가. 데이터 전처리 (Data Preprocessing)

* **타겟 변수 인코딩:** 예측 대상인 `Personality` 컬럼을 머신러닝 모델이 학습할 수 있도록 'Introvert'는 1, 'Extrovert'는 0으로 변환
* **결측치 처리:** 데이터의 특성을 고려하여 결측치 처리를 진행
    * `Friends_circle_size`, `Post_frequency`와 같이 연속적인 분포를 나타내는 컬럼은 **평균값(mean)**으로 대체
    * `Going_outside`, `Social_event_attendance` 등 개인의 성향을 나타내는 컬럼은 극단치의 영향을 덜 받는 **중앙값(median)**으로 대체
* **범주형 변수 처리:**
    * 'Yes'/'No' 형태의 `Stage_fear`, `Drained_after_socializing` 컬럼을 숫자 1/0으로 변환
    * 변환 후에도 남아있는 결측치는 가장 빈도가 높은 값인 **최빈값(mode)**으로 채워 데이터의 일관성을 유지
* **데이터 확장 (Data Augmentation):**
    * 모델의 학습 데이터 양을 늘리고 일반화 성능을 높이기 위해, 주어진 대회용 훈련 데이터 외에 **원본 데이터셋을 추가로 병합**하여 전체 훈련 데이터를 구성


### 나. 탐색적 데이터 분석 (EDA)
본격적인 모델링에 앞서, 변수 간의 관계를 파악하고 데이터의 숨겨진 구조를 발견하기 위해 탐색적 데이터 분석(EDA)을 수행

* **상관관계 분석 (Correlation Analysis):**
    * 분석 결과, 변수들은 성격 유형에 따라 크게 두 그룹으로 나뉘는 뚜렷한 경향을 보임
    * **내향성 연관 그룹:** `Time_spent_Alone`, `Stage_fear`, `Drained_after_socializing`
    * **외향성 연관 그룹:** `Social_event_attendance`, `Going_outside`, `Friends_circle_size`, `Post_frequency`
    * 이는 특정 행동 패턴들이 특정 성격 유형과 강하게 연결되어 있음을 시사하며, 이후 Feature Importance 결과와도 일맥상통하는 부분이 존재
  <img width="797" height="688" alt="image" src="https://github.com/user-attachments/assets/5bbcbb6a-80c1-45eb-ac30-6cfac5cadffb" />
* **차원 축소 실험 (PCA):**
    * 변수들을 더 적은 수의 주성분으로 압축하여 모델의 복잡도를 줄일 수 있는지 확인하기 위해 주성분 분석(PCA) 시도
    * 분석 결과, 4개의 주성분만으로 전체 데이터 분산의 **96%**를 설명할 수 있음을 확인하며 차원 축소의 가능성 발견
    * **하지만,** 이 주성분들을 실제 모델의 입력값으로 사용했을 때 **오히려 원본 변수들을 모두 사용한 모델보다 예측 정확도가 떨어지는 결과**가 나타남
    * 이를 통해 데이터셋은 **각 개별 변수가 가진 고유한 정보가 성향 예측에 매우 중요**하며, 인위적인 차원 축소가 오히려 정보 손실을 야기한다고 판단하여 최종 모델에는 PCA를 적용하지 않음
      
 * **PCA 바이플롯(Biplot) 시각화:**
    * 주성분 분석 결과를 바이플롯으로 시각화한 결과, 데이터 포인트들이 **PC1 축을 기준으로 명확하게 두 개의 군집(Cluster)으로 나뉘는 것**을 확인
    * 오른쪽 군집은 `Stage_fear`, `Drained_after_socializing` 변수와 같은 방향에 위치하며 **내향성 그룹**을, 왼쪽 군집은 `Friends_circle_size` 등 사회적 활동 변수들과 같은 방향에 위치하며 **외향성 그룹**으로 나타남
    * 이는 앞서 분석한 상관관계와 정확히 일치하는 결과로, **데이터가 두 성향에 따라 선형적으로 잘 구분되는 구조**임을 한번더 확인함
    ![PCA Biplot of Personality Data](https://github.com/kyungmin-DA/predict-the-introverts-extroverts-analysis/blob/main/attachment/pca_2d.png?raw=true)

 
### 다. 모델링 (Modeling)
단일 모델의 한계를 극복하고 예측 안정성과 정확도를 높이기 위해 **앙상블(Ensemble)** 기법을 사용

* **특성 스케일링 (Feature Scaling):**
    * 모델 훈련 전, `StandardScaler`를 이용해 모든 특성(feature)의 스케일 통일. 이를 통해 특정 변수가 모델에 과도한 영향을 미치는 것을 방지하고 안정적인 학습을 유도함
* **앙상블 모델 구축:**
    * 성능이 검증된 3개의 그래디언트 부스팅 모델(`CatBoost`, `XGBoost`, `LightGM`)을 기본 모델로 선정
    * `VotingClassifier`를 사용하여 세 모델의 예측 확률값을 종합적으로 고려하는 **소프트 보팅(Soft Voting)** 방식을 채택하여 최종 예측의 신뢰도를 높임

* **모델 성능 평가:**
   * 훈련된 모델의 객관적인 성능을 측정하기 위해, 전체 훈련 데이터의 20%를 검증용 데이터셋으로 분리하여 평가를 진행
   * 그 결과, **약 96%의 정확도(Accuracy)**와 **0.97의 ROC AUC 점수**를 달성하여, 모델이 내/외향인을 매우 높은 수준으로 판별함을 확인
       ### 검증 데이터셋 성능
       | 평가 지표 (Metric)      | 점수 (Score) |
       | :---------------------- | :----------: |
       | ✅ **정확도 (Accuracy)** |   **0.9579** |
       | ✅ **ROC AUC** |   **0.9715** |
       | F1-Score (Extrovert)    |     0.97     |
       | F1-Score (Introvert)    |     0.93     |
   
       ### Kaggle 최종 점수
       | Kaggle Score            | 점수 (Score) |
       | :---------------------- | :----------: |
       | 🏆 **Public Score** |   **0.9741** |
       | 🏆 **Private Score** |   **0.9676** |
   
    * 아래 Confusion Matrix에서 볼 수 있듯, 모델의 오분류 사례가 적은 수준으로 파악
      <img width="548" height="432" alt="image" src="https://github.com/user-attachments/assets/da9ce1a7-3c26-4fdf-a4d7-3d15b487ce34" />


* **핵심 예측 변수 확인 (Feature Importance):**
    * 앙상블 모델이 어떤 변수를 중요하게 보고 예측을 수행했는지 확인하기 위해 특성 중요도를 분석
    * 분석 결과, **'친구 수(Friends_circle_size)'**와 **'혼자 보내는 시간(Time_spent_Alone)'**이 내/외향인을 구분하는 가장 결정적인 변수로 나타남
    * 반면, '무대 공포증'이나 '사교 활동 후 피로감'과 같이 주관적인 감정 상태를 나타내는 변수들은 상대적으로 낮은 중요도를 기록
    
    <img width="1189" height="790" alt="image" src="https://github.com/user-attachments/assets/5612bd66-75cf-4b71-973a-26d137154953" />


## 3. 결론 및 인사이트

### 가. 프로젝트 요약
3개의 데이터셋을 통합하고, `CatBoost`, `XGBoost`, `LightGBM` 모델을 소프트 보팅 방식으로 앙상블하여 Kaggle Private Score 기준 **0.9676**의 높은 예측 성능을 달성. 특히 모델 분석 결과, 개인의 **객관적인 사회적 활동 패턴**이 주관적 감정 상태보다 성향을 더 잘 예측하는 핵심 지표임을 발견

### 나. 주요 발견점 (Key Findings)
* **행동은 감정보다 강하다:** 특성 중요도 분석 결과, '친구 수', '혼자 보내는 시간' 등 **관찰 가능한 행동 지표**가 예측에 가장 큰 영향을 미침. 반면 '무대 공포증' 등 **자기 보고식 감정 지표**의 영향력은 상대적으로 낮았는데, 이는 실제 행동 패턴이 성향을 더 강력하게 대변한다고 볼 수 있음
* **데이터 구조의 명확성:** PCA 바이플롯 시각화를 통해 데이터가 '내향성 연관 변수 그룹'과 '외향성 연관 변수 그룹'으로 명확히 나뉘는 선형적 구조를 가지고 있음을 시각적으로 확인

### 다. 개선 방안 및 고찰
* **클래스 불균형 해소:** 'Introvert' 클래스의 예측 성능이 근소하게 낮았던 점을 고려할 때, 향후 **SMOTE와 같은 오버샘플링 기법**으로 데이터 불균형을 해소한다면 소수 클래스의 예측 정확도를 더욱 높일 수 있을 것으로 예상
* **무분별한 차원 축소의 함정:** EDA 과정에서 시도한 PCA가 오히려 성능을 저하시키는 결과를 통해, 모든 변수가 모델에게 유의미한 정보를 제공하고 있으며 **인위적인 차원 축소가 항상 성능 향상으로 이어지지는 않는다는 결과**를 확인
* **선택적 차원 축소 및 파생 변수 생성:** EDA에서 발견한 '내향성 연관 그룹'(`Time_spent_Alone` 등)에만 선택적으로 PCA를 적용하여, 여러 변수를 대표하는 하나의 '내향성 지표'와 같은 파생 변수를 생성하는 전략을 시도해볼 수 있음. 이는 변수 간 다중공선성 문제를 완화하고 모델에게 더 정제된 신호를 제공하여 성능 향상에 기여했을 가능성 존재
  
## 4. 사용 기술
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, CatBoost, XGBoost, LightGBM, Matplotlib, Seaborn
* **Tool:** databricks(Notebook)

---
