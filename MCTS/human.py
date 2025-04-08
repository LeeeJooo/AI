class Human:
    def __init__(self, env, pid):
        self._env = env
        self.pid = pid

    def get_action(self):
        print("돌을 둘 좌표를 입력하세요. 예) 1,2")
        r, c = map(int, input().split(","))
        action = (r, c)

        is_available = self._env.check_available_action(action)

        if not is_available:
            print("다시 입력하세요")
        elif is_available:
            return action

            

    def __str__(self):
        return "Human {}".format(self.player)