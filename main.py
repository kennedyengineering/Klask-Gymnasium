import pygame
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE)

from klask_render import render_game_board

import Box2D
from Box2D.b2 import (world, polygonShape, circleShape, staticBody, dynamicBody)

# --- constants ---
# Box2D deals with meters, but we want to display pixels,
# so define a conversion factor:
# PPM = 20.0  # pixels per meter
PPM = 2000
TARGET_FPS = 60
TIME_STEP = 1.0 / TARGET_FPS
SCREEN_WIDTH, SCREEN_HEIGHT = 1000, 1000

# --- pygame setup ---
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
pygame.display.set_caption('Simple pygame example')
clock = pygame.time.Clock()

# --- pybox2d world setup ---
# Create the world
world = world(gravity=(0, -9.8), doSleep=True)

# And a static body to hold the ground shape
# ground_body = world.CreateStaticBody(
#     position=(0, 0),
#     shapes=polygonShape(box=(50, 1)),
# )

# # Create a couple dynamic bodies
# body0 = world.CreateStaticBody(position=(0,0))
# box0 = body0.CreatePolygonFixture(box=(32, 24), density=1, friction=0.3)

# body = world.CreateDynamicBody(position=(18.75, 20))
# circle = body.CreateCircleFixture(radius=0.5, density=1, friction=0.3)

# # Create joint
# joint = world.CreateFrictionJoint(bodyA=body0, bodyB=body, maxForce=7.5)

colors = {
    staticBody: (255, 255, 255, 255),
    dynamicBody: (127, 127, 127, 255),
}

# Let's play with extending the shape classes to draw for us.


def my_draw_polygon(polygon, body, fixture):
    vertices = [(body.transform * v) * PPM for v in polygon.vertices]
    vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
    pygame.draw.polygon(screen, colors[body.type], vertices)
polygonShape.draw = my_draw_polygon


def my_draw_circle(circle, body, fixture):
    position = body.transform * circle.pos * PPM
    position = (position[0], SCREEN_HEIGHT - position[1])
    pygame.draw.circle(screen, colors[body.type], [int(
        x) for x in position], int(circle.radius * PPM))
    # Note: Python 3.x will enforce that pygame get the integers it requests,
    #       and it will not convert from float.
circleShape.draw = my_draw_circle

# --- main game loop ---

running = True
while running:
    # Check the event queue
    for event in pygame.event.get():
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            # The user closed the window or pressed escape
            running = False

    screen.fill((0, 0, 0, 0))
    # Draw the world
    for body in world.bodies:
        for fixture in body.fixtures:
            fixture.shape.draw(body, fixture)

    render_game_board(PPM, screen)

    # Make Box2D simulate the physics of our world for one step.
    world.Step(TIME_STEP, 10, 10)

    # Flip the screen and try to keep at the target FPS
    pygame.display.flip()
    clock.tick(TARGET_FPS)

pygame.quit()
print('Done!')

