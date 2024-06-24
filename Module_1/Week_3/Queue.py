class MyQueue():
    def __init__(self, capacity):
        self.capacity = capacity
        self.queue = []

    def is_empty(self):
        return len(self.queue) == 0

    def is_full(self):
        return len(self.queue) == self.capacity

    def dequeue(self):
        if len(self.queue) != 0:
            return self.queue.pop(0)
        else:
            return 'Error! Queue is empty'

    def enqueue(self, value):
        if len(self.queue) != self.capacity:
            self.queue.append(value)
        else:
            return 'Error! Queue is full'

    def front(self):
        if len(self.queue) != 0:
            return self.queue[0]
        else:
            return 'Error! Queue is empty'

if __name__=="__main__":
    queue1 = MyQueue(capacity=5)
    queue1.enqueue(1)
    queue1.enqueue(2)

    print(queue1.is_full())
    print(queue1.front())
    print(queue1.dequeue())
    print(queue1.front())
    print(queue1.dequeue())
    print(queue1.is_empty())
