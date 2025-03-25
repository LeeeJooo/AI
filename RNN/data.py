# 파일 읽기
from config import *
from constants import *

# path에서 지정한 패턴에 일치하는 파일 및 디렉토리 목록을 리스트로 반환
def findFiles(path) :
    return glob.glob(path)

files = findFiles("data/names/*.txt")

# 각 언어의 이름 목록인 category_lines 사전 생성
category_lines = {}
all_categories = []
idx_to_category = {}
category_to_idx = {}

# 파일을 읽고 줄 단위로 분리
def readLines(filename):
    with open(filename, mode='r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]
    return lines

idx = 0
for file in files:
    filename = os.path.basename(file)           # 경로에서 파일명만 추출
    category = os.path.splitext(filename)[0]    # 파일명, 확장자 분리 후 파일명 추출
    all_categories.append(category)             # 파일명 -> 카테고리
    lines = readLines(file)                     # 파일명에 해당하는 내용 추출
    category_lines[category] = lines            # {파일명 : 내용}
    idx_to_category[idx] = category
    category_to_idx[category] = idx

n_categories = len(all_categories)


all_letters =  string.ascii_letters + '.,;'
n_letters = len(all_letters)

def letterToIndex(letter):
    return all_letters.find(letter)

def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# One-Hot 벡터로 변환
# shape : (line_len, 1, n_letters)
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for i, letter in enumerate(line):
        tensor[i][0][letterToIndex(letter)] = 1
    return tensor

def linesToTensors(lines):
    for i, line in enumerate(lines):
        lines[i] = lineToTensor(line)


# zip으로 묶어서 shuffle
# combined_train = list(zip(x_train, y_train))
# combined_eval = list(zip(x_eval, y_eval))
# combined_test = list(zip(x_test, y_test))
# random.shuffle(combined_train)
# random.shuffle(combined_eval)
# random.shuffle(combined_test)
# x_train, y_train = map(list, zip(*combined_train))
# x_eval, y_eval = zip(*combined_eval)
# x_test, y_test = zip(*combined_test)

# x_train = torch.tensor(x_train)
# y_train = torch.tensor(y_train)
# x_eval = torch.tensor(x_eval)
# y_eval = torch.tensor(y_eval)
# x_test = torch.tensor(x_test)
# y_test = torch.tensor(y_test)