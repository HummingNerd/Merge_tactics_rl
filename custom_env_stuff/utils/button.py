import pygame 

class Button():
    def __init__(self, x, y, image, size, name):
        self.image = pygame.transform.scale(image, ((int(size[0])), int(size[1])))
        self.rect = self.image.get_rect()
        self.rect.topleft = (x,y)
        self.clicked = False
        self.name = name

    def draw(self, surface):
        surface.blit(self.image, (self.rect.x, self.rect.y))
        action = False
        pos = pygame.mouse.get_pos()

        if self.rect.collidepoint(pos):
            if pygame.mouse.get_pressed()[0] == 1 and self.clicked == False:
                self.clicked = True
                action = True
        if pygame.mouse.get_pressed()[0] == 0:
            self.clicked = False
        return action