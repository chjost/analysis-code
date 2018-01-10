#!/usr/bin/python

def main():
    print("Hello World")
if __name__=="__main__":
    try:
        main()
    except(KeyboardInterrupt):
        print("KeyboardInterrupt")
