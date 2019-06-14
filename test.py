import pygame, math,random

x=[0.0]*5
y=[0.0]*5
random.seed()
for val in range(0,len(x)):
    x[val]=random.random()
for val in range(0,len(y)):
    y[val]=random.random()
print(x,y)
screen=pygame.display.set_mode((500,500))
pygame.display.set_caption("Scary port v4084")
while True:
    screen.fill((255,255,255))
    for index in range(0,len(x)):
        pygame.draw.circle(screen,(0,0,0),(int(200*(x[index])),int(200*(y[index]))),2,0)
        for other in range(0,len(x)):
            total=0
            if other!=index:
                total+=(x[index]-x[other])**2
                total+=(y[index]-y[other])**2
                total=(total)
                x[index]=x[index]+.000001*(x[index]-x[other])/total
                y[index]=y[index]+.000001*(y[index]-y[other])/total
                if x[index]>1:
                    x[index]=1
                if y[index]>1:
                    y[index]=1
                if x[index]<0:
                    x[index]=0
                if y[index]<0:
                    y[index]=0
                # print("happens",x[index]-x[other])
    pygame.display.flip()
    print("flippin")
