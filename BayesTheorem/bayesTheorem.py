'''
pandas와 numpy를 활용하여 확률 테이블 생성 및 계산
scipy.stats를 이용한 베타 분포 적용 실습
'''

import pandas as pd
import numpy as np

def bayes_theorem(p_B_given_A, p_A, p_B):
    '''
    P(A|B) = P(B|A)*P(A)/P(B)
    '''
    return p_B_given_A*p_A/p_B

# target 변수
# 어떤 값을 기준으로...
# IG -> 
# entrop = 1 - sum p_i^2
# 100 20 -> 0.2
# 20, 50 , 30 -> 
# 0 : 1, 1 : 19
# 1/20 * 0.2 >> 곱한다는 의미 P(A|B)
# 특정 기준 A(Pclass:1)에 대해 클래스 B(Survived:0, 사망)가 선택될 확률

'''
node split : IG 가 가장 큰 Feature, 혹은 Gini Impurity가 가장 작은 Feature
Threshold :
  >> 수치형 데이터일 경우 : 모든  unique한 values 사이의 평균을 계산한 후 각각의 값을 threshold value로 두고 split 했을 때의 불순도 감소량 관찰
  >> 범주형 데이터일 경우 : 모든 values를 threshold value로 두고 split했을 때의 불순도 감소량 관찰
'''