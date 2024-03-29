#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import sys, select, termios, tty

settings = termios.tcgetattr(sys.stdin)

def getKey():
	tty.setraw(sys.stdin.fileno())
	select.select([sys.stdin], [], [], 0)
	key = sys.stdin.read(1)
	termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
	return key

def main(args=None):
    if args is None:
         args = sys.argv
        
    rclpy.init(args=args)
    node = rclpy.create_node('enter_key_publisher')
    pub = node.create_publisher(String,'/save_images',35)
    try:
        while(1):
             key = getKey()
             if(key == ' '):
                msg = String()
                msg.data = ''
                pub.publish(msg)
             elif(key == 'q'):
                node.destroy_node()
                rclpy.shutdown()
                exit()
    except:
        print("Exception")
    finally:
         termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
         


if __name__ == '__main__':
    main()
