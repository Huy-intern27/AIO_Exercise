class MyStack():
    def __init__(self, capacity):
        self.capacity = capacity
        self.stack = []

    def is_empty(self):
        return len(self.stack) == 0

    def is_full(self):
        return len(self.stack) == self.capacity

    def pop(self):
        if len(self.stack) != 0:
            return self.stack.pop(0)
        else:
            return 'Error! Stack is empty'

    def push(self, value):
        if len(self.stack) != self.capacity:
            self.stack.insert(0, value)
        else:
            return 'Error! Stack is full'

    def top(self):
        if len(self.stack) != 0:
            return self.stack[0]
        else:
            return 'Error! Stack is empty'

if __name__=="__main__":
    stack1 = MyStack(capacity=5)
    stack1.push(1)
    stack1.push(2)

    print(stack1.is_full())
    print(stack1.top())
    print(stack1.pop())
    print(stack1.top())
    print(stack1.pop())
    print(stack1.is_empty())
