from rs4 import app
import sys
new_limit = 10000000

if __name__ == '__main__':
    print("hello")
    sys.setrecursionlimit(new_limit)
    app.run(debug=True)
