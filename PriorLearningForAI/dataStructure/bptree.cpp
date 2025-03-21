#include <vector>
#include <iostream>

using namespace std;

struct Node {
    bool isLeaf;
    vector<int> keys;
    vector<Node*> children;
    Node* next;
    Node(bool isLeaf) : isLeaf(isLeaf), next(nullptr) {}

    bool hasKey(int value) {
        int i=0;
        while(i<keys.size() && keys[i] < value) {
            i++;
        }
        if (keys[i]==value) {
            return true;
        }
        return false;
    }
    
    int partitionKeys(int start, int end) {
        if (start > end) {
            return 0;
        }

        int pivot = keys[end];
        int i = start-1;

        for (int j=start;j<end;j++) {
            if (keys[j] < pivot) {
                i++;
                int tmp = keys[i];
                keys[i] = keys[j];
                keys[j] = tmp;
            }
        }

        int tmp = keys[i+1];
        keys[i+1] = keys[end];
        keys[end] = tmp;

        return i+1;
    }

    void sortKeys() {
        if (keys.size() > 1) {
            int start = 0;
            int end = keys.size()-1;
            int pi = partitionKeys(start, end);
            partitionKeys(start, pi-1);
            partitionKeys(pi+1, end);
        }
    }

    int partitionChildren(int start, int end) {
        int pivot = children[end]->keys[0];
        int i = start-1;

        for (int j=start;j<end;j++) {
            if (children[j]->keys[0] < pivot) {
                i++;
                Node* tmp = children[i];
                children[i] = children[j];
                children[j] = tmp;
            }
        }

        Node* tmp = children[i+1];
        children[i+1] = children[end];
        children[end] = tmp;

        return i+1;
    }

    void sortChildren() {
        if (!children.empty()) {
            int start = 0;
            int end = children.size()-1;

            int pi = partitionChildren(start, end);
            partitionChildren(start, pi-1);
            partitionChildren(pi+1, end);
        }
    }
};

class BPTree {
private:
    const int N = 4;
    Node* root;

    int findIndex(Node* curNode, int value) {
        int i = 0;
        while (i<curNode->keys.size() && curNode->keys[i] <= value) {
            i++;
        }
        return i;
    }

    int findLeafIndex(Node* curNode, int value) {
        int i = 0;
        while (i<curNode->keys.size() && curNode->keys[i] < value) {
            i++;
        }
        return i;
    }

    void splitLeafNode(Node* parent, Node* curNode) {
        int midIdx = curNode->keys.size()/2;
        int midVal = curNode->keys[midIdx];

        Node* newLeaf = new Node(true);

        int i=0;
        while (i<midIdx) {
            newLeaf->keys.push_back(curNode->keys[0]);  // 추가
            curNode->keys.erase(curNode->keys.begin()); // 삭제
            i++;
        }

        if (parent==nullptr) {
            Node* newNode = new Node(false);
            newNode->keys.push_back(midVal);
            newNode->children.push_back(newLeaf);
            newNode->children.push_back(curNode);
            root = newNode;
        }
        else {
            parent->keys.push_back(midVal);
            parent->sortKeys();
            parent->children.push_back(newLeaf);
            parent->sortChildren();
        }
    }

    void splitMidNode(Node* parent, Node* curNode) {
        if (curNode->isLeaf) {
            return;
        }
        
        for (int i=0; i<curNode->children.size(); i++) {
            splitMidNode(curNode, curNode->children[i]);
        }

        if (curNode->keys.size() > N) {
            int midIdx = curNode->keys.size()/2;
            int midVal = curNode->keys[midIdx];
            
            Node* newMidNode = new Node(false);
            for (int i=0;i<midIdx;i++) {
                newMidNode->keys.push_back(curNode->keys[0]);
                curNode->keys.erase(curNode->keys.begin());
                newMidNode->children.push_back(curNode->children[0]);
                curNode->children.erase(curNode->children.begin());
            }
            newMidNode->children.push_back(curNode->children[0]);
            curNode->children.erase(curNode->children.begin());
            curNode->keys.erase(curNode->keys.begin());

            if (parent==nullptr) {
                Node* newRoot = new Node(false);
                newRoot->keys.push_back(midVal);
                newRoot->children.push_back(newMidNode);
                newRoot->children.push_back(curNode);
                root = newRoot;
            }
            else {
                parent->keys.push_back(midVal);
                parent->sortKeys();
                parent->children.push_back(newMidNode);
                parent->sortChildren();
            }
        }
    }

    void print(int depth, Node* curNode, int idx) {
        if (!curNode->isLeaf) {
            for (int i=0; i<curNode->children.size(); i++) {
                print(depth+1, curNode->children[i], i);
            }
        }

        cout << " >> DEPTH " << depth << " | " << idx << "번째 NODE : ";
        for (int i=0; i<curNode->keys.size(); i++) {
            cout << curNode->keys.at(i) << " "; 
        }
        cout << endl;
    }

    bool search(int value) {
        if (root==nullptr) {
            cout << "[SEARCH] " << value << "는 없습니다." << endl;
            return false;
        }

        int cnt=0;
        Node* curNode = root;
        while (!curNode->isLeaf) {
            int idx = findIndex(curNode, value);
            curNode = curNode->children[idx];
        }

        int data_idx = 0;
        while (data_idx<curNode->keys.size() && curNode->keys[data_idx] < value) {
            data_idx++;
        }
        int data_value = curNode->keys[data_idx];
        if (data_value==value) {
            return true;
        }
        else {
            return false;
        }
    }

public:
    BPTree() : root(nullptr) {}

    void search_data(int value) {
        if (search(value)) {
            cout << "[SEARCH]" << value << "을 찾았습니다." << endl;
        }
        else {
            cout << "[SEARCH]" << value << "은 없습니다." << endl;
        }
    }
    
    void insert_data(int value) {
        // bptree가 비어있을 경우
        if (root==nullptr) {
            Node* newNode = new Node(true);
            newNode->keys.push_back(value);
            root = newNode;
            cout << "[INSERT] " << value << " 완료" << endl;
            return;
        }
        
        // leaf 노드까지 내려감
        Node* curNode = root;
        Node* parent = nullptr;
        int parentKeyIdx = -1;
        while (!curNode->isLeaf) {
            parent = curNode;
            int parentKeyIdx = findIndex(curNode, value);
            curNode = curNode->children[parentKeyIdx];
        }

        int insertIdx = findIndex(curNode, value);

        // 중복 확인
        if (insertIdx<curNode->keys.size() && curNode->keys[insertIdx] == value) {
            cout << "[INSERT] 중복 : " << value <<"는 이미 존재합니다." << endl;
            return;
        }

        // Insert data
        curNode->keys.push_back(value);
        curNode->sortKeys();

        // Leaf Node 공간 넉넉
        if (curNode->keys.size() <= N) {
            if (parent!=nullptr && parentKeyIdx>0) {
                parent->keys.at(parentKeyIdx-1) = curNode->keys.front();
            }
            cout << "[INSERT] " << value << " 완료" << endl;
            return;
        }
        
        // Split Leaf Node
        splitLeafNode(parent, curNode);

        // // 중간 노드 split
        splitMidNode(nullptr, root);

        cout << "[INSERT] " << value << " 완료" << endl;
    }

    void delete_data(int value) {
        if (root==nullptr) {
            cout << "[DELETE] 실패 : " << value << "은 없습니다." << endl;
            return;
        }
        
        Node* parent = nullptr;
        Node* curNode = root;
        int parentKeyIdx = -1;
        while (!curNode->isLeaf) {
            parent = curNode;
            parentKeyIdx = findIndex(curNode, value);
            curNode = curNode->children[parentKeyIdx];
        }
        
        int dataIdx = findLeafIndex(curNode, value);
        int dataValue = curNode->keys[dataIdx];
        
        if (dataValue!=value) {
            cout << "[DELETE] 실패 : " << value << "은 없습니다." << endl;
            return;
        } 
        
        // data 삭제
        curNode->keys.erase(curNode->keys.begin() + dataIdx);
        if (parent==nullptr) {
            if (curNode->keys.size()==0) {
                root=nullptr;
            }
            cout << "[DELETE] 성공 : " << value << "을 삭제했습니다." << endl;
            return;
        }
        // key로 존재하면 삭제
        bool hasKey = parent->hasKey(value);
        if (hasKey) {
            int idx= findLeafIndex(parent, value);
            parent->keys.erase(parent->keys.begin()+idx);
        }
        
        // 최소 개수 충족 시 종료
        if (curNode->keys.size()>=N/2) {
            if (hasKey) {
                parent->keys.push_back(curNode->keys[0]);
            }
            cout << "[DELETE] 성공 : " << value << "을 삭제했습니다." << endl;
            return;
        }
        
        int dataNeededCnt = N/2-curNode->keys.size();
        
        Node* leftNode = nullptr;
        Node* rightNode = nullptr;
        
        // 형제 노드 빌리기 : 왼쪽
        if (parentKeyIdx-1 >=0) {
            leftNode = parent->children[parentKeyIdx-1];
            int leftNodeSize = leftNode->keys.size();
            
            if (leftNodeSize-dataNeededCnt>=N/2) {
                int i=0;
                while(i<dataNeededCnt) {
                    curNode->keys.push_back(leftNode->keys.back());
                    leftNode->keys.erase(leftNode->keys.begin() + leftNode->keys.size()-1);
                    i++;
                }
                curNode->sortKeys();
                
                parent->keys.push_back(curNode->keys.front());
                parent->sortKeys();
                
                cout << "[DELETE] 성공 : " << value << "을 삭제했습니다." << endl;
                return;
            }
        }

        // 형제 노드 빌리기 : 오른쪽
        if (parentKeyIdx+1 < parent->children.size()) {
            rightNode = parent->children[parentKeyIdx+1];
            int rightNodeSize = rightNode->keys.size();
            
            if (rightNodeSize-dataNeededCnt>=N/2) {
                int rightNodeKeyRemoved = rightNode->keys[0];
                int rightNodeKeyRemovedIdx = findLeafIndex(parent, rightNodeKeyRemoved);
                parent->keys.erase(parent->keys.begin() + rightNodeKeyRemovedIdx);

                int rightNodeKeyAdded = rightNode->keys[1];
                parent->keys.push_back(rightNodeKeyAdded);

                int i=0;
                while(i<dataNeededCnt) {
                    curNode->keys.push_back(rightNode->keys.front());
                    rightNode->keys.erase(rightNode->keys.begin());
                    i++;
                }
                if (parentKeyIdx>0) {
                    parent->keys.push_back(curNode->keys.front());
                }
                parent->sortKeys();
                
                cout << "[DELETE] 성공 : " << value << "을 삭제했습니다." << endl;
                return;
            }
        }
        
        // 병합 : 왼쪽 노드
        if (leftNode!=nullptr) {
            int i = 0;
            while (i<curNode->keys.size()) {
                leftNode->keys.push_back(curNode->keys.front());
                curNode->keys.erase(curNode->keys.begin());
            }
            parent->children.erase(parent->children.begin()+parentKeyIdx);
            cout << "[DELETE] 성공 : " << value << "을 삭제했습니다." << endl;
            return;
        }
        
        // 병합 : 오른쪽 노드
        if (rightNode!=nullptr) {
            int i = 0;
            while (i<curNode->keys.size()) {
                rightNode->keys.push_back(curNode->keys.back());
                curNode->keys.erase(curNode->keys.begin()+curNode->keys.size()-1);
            }
            parent->children.erase(parent->children.begin()+parentKeyIdx);
            cout << "[DELETE] 성공 : " << value << "을 삭제했습니다." << endl;
            return;
        }
    }

    void print_bptree() {
        if (root==nullptr) {
            cout << "[PRINT] NULL" << endl;
            return;
        }
        int depth = 0;
        cout << "[PRINT] B Plus Tree" << endl;
        print(depth, root, 0);
    }

    ~BPTree() {}
};

int main() {
    BPTree bptree = BPTree();
    bptree.insert_data(1);
    bptree.print_bptree();
    bptree.search_data(1);
    bptree.delete_data(1);
    bptree.print_bptree();
    // for (int i=0; i<=13; i++) {
    //     if (i==1) {
    //         continue;
    //     }
    //     bptree.insert_data(i);
    //     bptree.print_bptree();
    // }
    // for (int i=-2; i<5; i++) {
    //     bptree.search_data(i);
    // }
    // bptree.delete_data(9);
    // bptree.print_bptree();

    return 0;
}