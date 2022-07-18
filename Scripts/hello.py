import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
"""
베이지안 정리이란 두 확률 변수의 사전확률과 사후 확률 사이의 관계를 나타내는 정리 
사전확률[prior probability] P(A)과 우도확률[likelihood probability] P(B|A)를 안다면 사후확률[posterior probability] P(A|B)를 알 수 있다.
사전확률[prior probability] P(A) : 결과가 나타나기 전에 결정되어 있는 A(원인)의 확률
우도확률[likelihood probability] P(B|A) : A(원인)가 발생하였다는 조건하에서 B(결과)가 발생할 확률
사후확률[posterior probability] P(A|B) : B(결과)가 발생하였다는 조건하에서 A(원인)가 발생하였을 확률

"""


"""
필터는 출력으로써 더 정확한 추정값을 얻기 위해 입력의 노이즈(불확실성)을 제거하는 것입니다.
베이지안 필터는 베이지안 통계가 적용된 필터입니다. 로봇은 0에서 9까지의 복도에 존재하고 움직입니다.
복도는 원형 복도 입니다. 9 다음에 0 입니다. 그리고 조명은 1, 3, 7번 위치에만 존재합니다.

"""

# 초기에는 로봇의 위치에 대한 정보가 없습니다. 그래서 로봇은 어느 위치에서나 존재 할 수 있으므로 모든 위치에 같은 확률을 할당합니다. 
grid_map = 10
belief = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

# 조명의 위치에 따른 확률
light = [0, 1, 0, 1, 0, 0, 0, 1, 0, 0]

# correct step : 센서 데이터를 사용하여 센서 데이터와 환경의 지도에 기반하여 추정된 상태를 수정 하는 단계 / 측정과 관측은 같은 의미
"""
처음에 로봇의 위치에 대한 정보가 없으므로 belief를 모두 같게 설정 하였습니다. belief = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
로봇이 빛을 감지하면 로봇의 위치에 대한 belief가 다음과 같이 변화 합니다. belief = [0, 0.333, 0, 0.333, 0, 0, 0, 0.333, 0, 0]
왜냐하면 우리는 지도에 조명이 3개 뿐이라는 것을 알고 조명이 1, 3, 7번 위치에 있다는 것을 알기 때문입니다. light = [0, 1, 0, 1, 0, 0, 0, 1, 0, 0]
하지만 로봇은 1, 3, 7번 위치 중 어디에 있는지는 모르고 정확한 정보가 없으므로 모든 조명에 같은 확률로 할당을 합니다. 그래서 belief가 belief = [0, 0.333, 0, 0.333, 0, 0, 0, 0.333, 0, 0]와 같이 된 것입니다.

"""

"""
하지만 correct step 즉 센서를 이용하는 측정 과정은 정확하지 않습니다. 왜냐하면 센서들은 완벽하지 않아 노이즈가 섞인 부정확한 데이터를 주기 때문입니다.
센서들의 감지 거리 및 해상도 분해능 등은 제한적이고 여러 외부 요소(환경)들에 의해 노이즈가 생겨 잘 측정이 안될 수도 있기 때문입니다.
우리는 여기서 우도(likelihood)를 사용합니다. 로봇이 움직였을때 센서가 측정되어 값이 어떻게 들어올 확률
빛이 감지된 위치를 나타내는, 노이즈가 있는 센서 데이터를 받은 후 로봇이 있을 수 있는 위치를 나타내는 벡터 likelihood = [1, 3, 1, 3, 1, 1, 1, 3, 1, 1]

"""
sensor_env = [1,3,1,3,1,1,1,3,1,1]

"""
환경에 대한 지도 envrionment_map
센서 데이터 z (z = 1 : 빛이 감지된 경우 / z = 0 : 빛이 감지되지 않은 경우)
센서 데이터의 정확도 z_prob : 센서가 빛을 감지한 gird에 할당된 값을 hold 합니다. 센서의 정확도가 높을 수록 해당 값이 잘 유지 됩니다. [ex) 해당 그리드 값 3 센서 정확도 90%이면 2.7]
어두운 영역의 grid cells의 값은 (1 - z_prob)이 곱해지게 됩니다. 이는 센서가 완벽하지 않다는 것을 나타내며, 따라서 센서가 어두운 곳에서도 빛을 감지할 수도 있기 때문에 확률값에 0을 할당하지 않습니다.
위의 함수는 환경 지도의 길이만큼 1로 초기화된 list를 만듭니다. 그 후, 각각의 cell을 업데이트 할 때,
지도와 센서의 데이터가 매치되면(센서의 값과 실제 지도에서 빛의 유무의 값을 비교) z_porb 값을 곱하며 매치되지 않으면 (1 - z_prob)를 곱합니다.
    
"""
def likelihood(environment_map, z, z_prob):
    likelihood = [1] * (len(environment_map))
    for index, grid_value in enumerate(environment_map):
        if grid_value == z:
            likelihood[index] *= z_prob
        else:
            likelihood[index] *= (1 - z_prob)
    return likelihood


"""
Normalization(정규화)
위에서 정의한 likelihood 벡터는 확률 분포가 아니기 때문에 확률 분포 함수로  바꿔져야 합니다. 그래서 각 값의 합이 1이 되도록 고정된 scale factor를 곱하여 스케일을 조절해 줘야 합니다.
변환 된 값들의 상대적인 차이는 원래의 값과 동일해야합니다.

normalizer = 1 / (likelihood vector의 모든 원소들의 합)
likelihood 벡터의 각각의 값에 normalizer를 곱합니다
likelihood = [1,3,1,3,1,1,1,3,1,1] 를 정규화한다고 하면, 다음과 같습니다.
모든 원소의 합 : 1+3+1+3+1+1+1+3+1+1 = 16  >> normalizer = 0.0625
따라서, normalized_likelihood = [0.0625,0.1875,0.0625,0.1875,0.0625,0.0625,0.0625,0.1875,0.0625,0.0625]    
"""
def normalize(inputList):
    """ calculate the normalizer, using: (1 / (sum of all elements in list)) """
    normalizer = 1 / float(sum(inputList))
    # multiply each item by the normalizer
    inputListNormalized = [x * normalizer for x in inputList]
    return inputListNormalized

"""
correct step은 현재의 belief를 수정하기 위해 센서의 측정값을 사용합니다. 수학적으로 말해서,
correct step은 현재의 belief를 likelihood function에 곱하는 것으로 구성됩니다.
즉, 각각의 belief를 각각의 likelihood의 원소에 곱하는 것입니다.
우리는 likelihood가 측정값과 환경의 지도를 비교하여 만들어졌다는 것을 알고있습니다.
추가적으로, correct step은 가능한 값들의 합이 1이 되도록 적절한 확률 분포를 만들기 위해 업데이트된 belief를 정규화 합니다.  

"""
def correct_step(likelihood, belief):
    output = []
    # element-wise multiplication (likelihood * belief)
    for i in range(0, len(likelihood)):
        output.append(likelihood[i]*belief[i])
    return normalize(output)


# 바 그래프 설정
x = np.arange(grid_map)
grid = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
plt.bar(x, belief,width=0.4)
plt.xticks(x, grid)

# x, y축 라벨 표시 및 y축 리미트 설정 (왜 1이냐 확률 스케일 상 1이기 때문)
plt.xlabel('move grid')
plt.ylabel('belif')
plt.ylim([0, 1])

# vscode에서 그래프 플롯할때 필수적인 거임
plt.show()


#x = np.array([[1, 2],
#             [3, 4]])
#np.linspace(0, 2, 20)
#np.arange(0, 2, 0.1)
#print('zeros\n', np.zeros(7))
#print('\nzeros(3x2)\n', np.zeros((3, 2)))
#print('\neye\n', np.eye(3))
#print('transpose\n', x.T)
#print('\nNumPy ninverse\n', np.linalg.inv(x))
#print('\nSciPy inverse\n', linalg.inv(x))
