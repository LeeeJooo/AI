#include <iostream>

struct Node {
    int data;
    Node* next;
    Node* before;
    Node(int value) : data(value), next(nullptr), before(nullptr) {}
};

class Queue {
private:
    Node* front;
    Node* back;
    int size;

public:
    Queue() : front(nullptr), back(nullptr), size(0) {
        std::cout<<"\n<< QUEUE IS CREATED >>"<<std::endl;
    }
    
    bool isEmpty(){
        return front == nullptr;
    }
    
    void push(int value) {
        Node* newNode = new Node(value);

        if (front==nullptr) {
            front=back=newNode;
        }
        else {
            back->next = newNode;
            back = newNode;
        }

        size++;

        std::cout <<"push " << value << std::endl;
    }

    void pop() {
        if (isEmpty()) {
            std::cout << "[POP] Queue is EMPTY" << std::endl;
            return;
        }
        Node* tmp = front;
        front = front->next;
        if (front==nullptr) {
            back = nullptr;
        }

        size--;
        
        std::cout << "POP " << tmp->data << std::endl;
        delete tmp;
    }

    void getSize(){
        std::cout << "Queue size is " << size << std::endl;
    }

    void getFront() {
        if (isEmpty()) {
            std::cout << "[GET FRONT] Queue is EMPTY" << std::endl;
            return;
        }
        std::cout << "Front Node is " << front->data << std::endl;
    }
    void getBack() {
        if (isEmpty()) {
            std::cout << "[GET BACK] Queue is EMPTY" << std::endl;
            return;
        }
        std::cout << "Back Node is " << back->data << std::endl;
    }

    void print() {
        Node* curNode = front;
        std::cout<<"[PRINT] ";
        while (curNode) {
            std::cout << curNode->data << " -> ";
            curNode = curNode->next;
        }
        std::cout << "NULL\n"; 
    }

    ~Queue() {
        while(!isEmpty()){
            pop();
        }
    }
};

class Stack {
private:
    Node* front;
    Node* back;
    int size;
    
public:
    Stack() : front(nullptr), back(nullptr), size(0) {
        std::cout<<"\n<< STACK IS CREATED >>"<<std::endl;
    }
    
    bool isEmpty(){
        return front == nullptr;
    }
    
    void push(int value) {
        Node* newNode = new Node(value);

        if (front==nullptr) {
            front=back=newNode;
        }
        else {
            newNode->before = back;
            back = newNode;
        }

        size++;

        std::cout <<"push " << value << std::endl;
    }

    void pop() {
        if (isEmpty()) {
            std::cout << "[POP] Stack is EMPTY" << std::endl;
            return;
        }
        Node* tmp = back;
        back = back->before;
        if (back==nullptr) {
            front = nullptr;
        }

        size--;
        
        std::cout << "POP " << tmp->data << std::endl;
        delete tmp;
    }

    void getSize(){
        std::cout << "Stack size is " << size << std::endl;
    }

    void getFront() {
        if (isEmpty()) {
            std::cout << "[GET FRONT] Stack is EMPTY" << std::endl;
            return;
        }
        std::cout << "Front Node is " << front->data << std::endl;
    }
    void getBack() {
        if (isEmpty()) {
            std::cout << "[GET BACK] Stack is EMPTY" << std::endl;
            return;
        }
        std::cout << "Back Node is " << back->data << std::endl;
    }
    
    void print() {
        Node* curNode = back;
        std::cout<<"[PRINT] ";
        while (curNode) {
            std::cout << curNode->data << " <- ";
            curNode = curNode->before;
        }
        std::cout << "NULL\n"; 
    }

    ~Stack() {
        while(!isEmpty()){
            pop();
        }
    }
};

class DoubleLinked {
private:
    Node* front;
    Node* back;
    int size;

public:
    DoubleLinked() : front(nullptr), back(nullptr), size(0) {
        std::cout << "\n<< DOUBLE LINKED LIST IS CREATED >>"<<std::endl;
    }

    bool isEmpty() {
        return front==nullptr;
    }

    void pushLeft(int value) {
        Node* newNode = new Node(value);

        if (front==nullptr){
            front=back=newNode;
        }
        else {
            newNode->next=front;
            front->before = newNode;
            front = newNode;
        }

        size++;

        std::cout << "Push Left " << value << std::endl;
    }
    
    void pushRight(int value) {
        Node* newNode = new Node(value);
        
        if (front==nullptr) {
            front=back=newNode;
        }
        else {
            back->next = newNode;
            newNode->before = back;
            back = newNode;
        }

        size++;

        std::cout << "Push Right " << value << std::endl;
    }
    
    void popLeft() {
        if (isEmpty()) {
            std::cout << "[POP LEFT] Double Linked List is EMPTY." << std::endl;
            return;
        }
        
        Node* tmp = front;
        front = front->next;

        if (front==nullptr) {
            back=nullptr;
        }

        size--;

        std::cout << "Pop Left " << tmp->data << std::endl;

        delete tmp;
    }
    
    void popRight() {
        if (isEmpty()) {
            std::cout << "[POP RIGHT] Double Linked List is EMPTY." << std::endl;
            return;
        }
        
        Node* tmp = back;
        back = back->before;

        if (back==nullptr) {
            front = nullptr;
        }
        
        size--;

        std::cout << "Pop Right " << tmp->data << std::endl;
        
        delete tmp;
    }
    
    void getSize(){
        std::cout << "Double Linked List size is " << size << std::endl;
    }

    void getFront() {
        if (isEmpty()) {
            std::cout << "[GET FRONT] Double Linked List is EMPTY." << std::endl;
            return;
        }
        
        std::cout << "Front is " << front->data << std::endl;
    }
    
    void getBack() {
        if (isEmpty()) {
            std::cout << "[GET BACK] Double Linked List is EMPTY." << std::endl;
            return;
        }
        
        std::cout << "Back is " << back->data << std::endl;
    }
    
    void print() {
        Node* curNode = front;
        std::cout<<"[PRINT] ";
        while (curNode) {
            std::cout<<curNode->data<<" -> ";
            curNode = curNode->next;
        }
        std::cout<<"NULL"<<std::endl;
    }

    ~DoubleLinked() {
        while(!isEmpty()){
            popLeft();
        }
    }
};

int main() {
    Queue q = Queue();
    q.getFront();
    q.getBack();
    q.pop();
    q.push(1);
    q.push(2);
    q.push(3);
    q.getSize();
    q.print();
    q.getFront();
    q.getBack();
    q.pop();
    q.pop();
    q.pop();
    q.pop();
    q.getSize();
    
    Stack s = Stack();
    s.getFront();
    s.getBack();
    s.pop();
    s.push(1);
    s.push(2);
    s.push(3);
    s.push(4);
    s.getSize();
    s.print();
    s.getFront();
    s.getBack();
    s.pop();
    s.pop();
    s.pop();
    s.pop();
    s.pop();
    s.getSize();
    
    DoubleLinked d = DoubleLinked();
    d.getFront();
    d.getBack();
    d.popLeft();
    d.popRight();
    d.pushLeft(1);
    d.print();
    d.pushLeft(2);
    d.print();
    d.pushRight(3);
    d.print();
    d.pushRight(4);
    d.print();
    d.getSize();
    d.getFront();
    d.getBack();
    
    d.popLeft();
    d.print();
    d.getSize();
    
    d.popLeft();
    d.print();
    d.getSize();
    
    d.popLeft();
    d.print();
    d.getSize();
    
    d.popLeft();
    d.print();
    d.getSize();
    

    return 0;
}