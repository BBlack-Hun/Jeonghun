#REINFORCE
import gym
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Hyperparameters
learning_rate = 0.0005
gamma = 0.99

# nn.Module을 상속받았다. --> torch에서 상속받아서..
class Policy(nn.Module):
    #무조건 쓰는 코드 -> 초기화
    def __init__(self):
        super(Policy, self).__init__()
        self.data = []

        #모델을 생성하는 곳..!
        # 딥러닝 relu로 128차원까지 확장 다시, 2차원으로 축소 -> softmax로 확률로 만들어줌
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 =  nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 128)
        self.fc6 = nn.Linear(128, 128)
        self.fc7 = nn.Linear(128, 2)
        #모델을 학습시켜주는 optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    #relu를 이용한 2 layer fc
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.softmax(self.fc7(x), dim=0)
        return x


    # 튜플데이터를 리스트로 저장한다.
    def put_data(self, item):
        self.data.append(item)

    def train(self):
        R = 0
        self.optimizer.zero_grad()
        #데이터를 뒤에서 부터 본다. --> -1
        for r, prob in self.data[::-1]:
            R = r + R * gamma
            #LOSS를 본다. gradient absent
            loss = -prob * R
            # torch에서 제공하는 함수이다. auto diff -> gradient absent가 다 계산이 된다.
            loss.backward()
            #episode 갯수만큼 gradient를 업데이트 해준다.
        self.optimizer.step() #왜 오류가 뜨니???
        #데이터를 비워준다.
        self.data = []


def main():
    env = gym.make('CartPole-v1')
    pi = Policy()
    score = 0
    print_interval = 20

    for n_epi in range(10000):
        obs = env.reset()
        done = False

        while not done:
            #Cartpole환경에서 관측되는 데이터 4개를 tensor화 시킨다.
            obs = torch.tensor(obs, dtype=torch.float, requires_grad=True)
            #생성된 데이터를 policy에 적용시킨다.
            out = pi(obs)
            #out은 왼쪽 or 오른쪽으로 갈 확률이다.
            out = out.cuda()
            m = Categorical(out)
            #랜덤확률인 Categorical로 둘의 확률을 정한다.
            #그 두 확률중 하나를 sampling한다.
            action = m.sample()
            #action값은 0또는 1이다.
            obs_prime, r, done, info = env.step(action.item())
            pi.put_data((r,torch.log(out[action])))
            obs = obs_prime
            score += r

            if score / 20 > 450:
                env.render()

        pi.train()

        if n_epi% print_interval == 0 and n_epi !=0:
            print("# of episode : {}, Avg_score : {}".format(n_epi, score/print_interval))
            if score % 20 >= 490:
                sys.exit(0)
            score = 0.0

    env.close()


if __name__ == '__main__':
    main()