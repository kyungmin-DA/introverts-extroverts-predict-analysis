# 내향인과 외향인 예측 분석 프로젝트(kaggle playground, 2025)

## 1. 프로젝트 개요 (Introduction)

* **분석 배경 및 목적:**
    * 이 프로젝트는 Kaggle의 'Playground Series - S5E7' 참여를 위해 진행되었습니다.
    * 주어진 대회 데이터를 바탕으로, 응답자의 성격 유형(내향형/외향형)을 예측하는 머신러닝 모델을 개발하고 성능을 최적화하는 것을 목표로 합니다.
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
본격적인 모델링에 앞서, 변수 간의 관계를 파악하고 데이터의 숨겨진 구조를 발견하기 위해 탐색적 데이터 분석(EDA)을 수행했습니다.

* **상관관계 분석 (Correlation Analysis):**
    * 분석 결과, 변수들은 성격 유형에 따라 크게 두 그룹으로 나뉘는 뚜렷한 경향을 보였습니다.
    * **내향성 연관 그룹:** `Time_spent_Alone`, `Stage_fear`, `Drained_after_socializing`
    * **외향성 연관 그룹:** `Social_event_attendance`, `Going_outside`, `Friends_circle_size`, `Post_frequency`
    * 이는 특정 행동 패턴들이 특정 성격 유형과 강하게 연결되어 있음을 시사하며, 이후 Feature Importance 결과와도 일맥상통하는 부분이었습니다.

* **차원 축소 실험 (PCA):**
    * 변수들을 더 적은 수의 주성분으로 압축하여 모델의 복잡도를 줄일 수 있는지 확인하기 위해 주성분 분석(PCA)을 시도했습니다.
    * 분석 결과, 4개의 주성분만으로 전체 데이터 분산의 **96%**를 설명할 수 있음을 확인하며 차원 축소의 가능성을 발견했습니다.
    * **하지만,** 이 주성분들을 실제 모델의 입력값으로 사용했을 때 **오히려 원본 변수들을 모두 사용한 모델보다 예측 정확도가 떨어지는 결과**를 보였습니다.
    * 이를 통해 우리 데이터셋은 **각 개별 변수가 가진 고유한 정보가 성향 예측에 매우 중요**하며, 인위적인 차원 축소가 오히려 정보 손실을 야기한다고 판단하여 최종 모델에는 PCA를 적용하지 않는 것으      로 결정했습니다.
    * 
 * **PCA 바이플롯(Biplot) 시각화:**
    * 주성분 분석 결과를 바이플롯으로 시각화한 결과, 데이터 포인트들이 **PC1 축을 기준으로 명확하게 두 개의 군집(Cluster)으로 나뉘는 것**을 확인할 수 있었습니다.
    * 오른쪽 군집은 `Stage_fear`, `Drained_after_socializing` 변수와 같은 방향에 위치하며 **내향성 그룹**을, 왼쪽 군집은 `Friends_circle_size` 등 사회적 활동 변수들과 같은 방향에 위치하며 **외향성 그룹**을 나타냅니다.
    * 이는 앞서 분석한 상관관계와 정확히 일치하는 결과로, **데이터가 두 성향에 따라 선형적으로 잘 구분되는 구조**임을 시각적으로 증명합니다.

    ![PCA Biplot of Personality Data](이미지_파일_경로.png)

 

### 다. 모델링 (Modeling)
단일 모델의 한계를 극복하고 예측 안정성과 정확도를 높이기 위해 **앙상블(Ensemble)** 기법을 핵심 전략으로 사용했습니다.

* **특성 스케일링 (Feature Scaling):**
    * 모델 훈련 전, `StandardScaler`를 이용해 모든 특성(feature)의 스케일을 통일했습니다. 이를 통해 특정 변수가 모델에 과도한 영향을 미치는 것을 방지하고 안정적인 학습을 유도했습니다.
* **앙상블 모델 구축:**
    * 성능이 검증된 3개의 그래디언트 부스팅 모델(`CatBoost`, `XGBoost`, `LightGM`)을 기본 모델로 선정했습니다.
    * `VotingClassifier`를 사용하여 세 모델의 예측 확률값을 종합적으로 고려하는 **소프트 보팅(Soft Voting)** 방식을 채택하여 최종 예측의 신뢰도를 높였습니다.

* **모델 성능 평가:**
    * 훈련된 모델의 객관적인 성능을 측정하기 위해, 전체 훈련 데이터의 20%를 검증용 데이터셋으로 분리하여 평가를 진행했습니다.
    * 그 결과, **약 96%의 정확도(Accuracy)**와 **0.97의 ROC AUC 점수**를 달성하여, 모델이 내/외향인을 매우 높은 수준으로 판별함을 확인했습니다.
    * 아래 혼동 행렬(Confusion Matrix)에서 볼 수 있듯, 모델의 오분류 사례가 매우 적은 것을 직관적으로 파악할 수 있습니다.
![Confusion Matrix for Personality Prediction](<img width="548" height="432" alt="image" src="https://github.com/user-attachments/assets/e4009a69-890b-4c30-87ec-0cb68b720621" />)


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
