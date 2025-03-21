#include <iostream>
#include <vector>

using namespace std;

int main() {
    vector<int> v;
    vector<int> v2;
    for (int i=0; i<3; i++) {
        v.push_back(i);
    }

    v2.assign(v.end()-2, v.end());
    for (int i=0; i<v.size(); i++) {
        cout << v[i] << " ";
    }
    cout << endl;
    for (int i=0; i<v2.size(); i++) {
        cout << v2[i] << " ";
    }
    
    cout << endl;
    
    v.erase(v.begin(), v.begin()+2);
    
    for (int i=0; i<v.size(); i++) {
        cout << v[i] << " ";
    }
    cout << endl;
    vector<int> v3;
    v3.push_back(100000);
    v3.push_back(100001);
    v2.insert(v2.end(), v3.begin(), v3.end());
    v2.insert(v2.begin()+2, 4);
    for (int i=0; i<v2.size(); i++) {
        cout << v2[i] << " ";
    }

}