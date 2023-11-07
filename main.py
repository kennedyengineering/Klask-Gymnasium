from dataclasses import dataclass
from random import choice
from math import dist

import pygame
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE, MOUSEBUTTONDOWN, MOUSEBUTTONUP, MOUSEMOTION)

import Box2D
from Box2D.b2 import (world, edgeShape, vec2, pi, contactListener)

from klask_render import render_game_board
from klask_constants import *

# TODO: prevent force from ball applied on puck, but keep force of puck on ball?

# --- constants ---
LENGTH_SCALER = 100         # Box2D doesn't simulate small objects very well. Scale distances into the meter range.
PPM = 20                    # Pixels per meter
TARGET_FPS = 120
TIME_STEP = 1.0 / TARGET_FPS
SCREEN_WIDTH, SCREEN_HEIGHT = KG_BOARD_WIDTH * PPM * LENGTH_SCALER, KG_BOARD_HEIGHT * PPM * LENGTH_SCALER

# --- pygame setup ---
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
pygame.display.set_caption('Klask Simulator')
clock = pygame.time.Clock()

# --- create contact listener ---
class myContactListener(contactListener):
    def __init__(self):
        contactListener.__init__(self)

        self.collision_list = []

    def PreSolve(self, contact, oldManifold):
        # Change the characteristics of the contact before the collision response is calculated

        # Check if a collision with a static body
        if contact.fixtureA.userData is None or contact.fixtureB.userData is None:
            return
        
        names = {contact.fixtureA.userData.name : contact.fixtureA, contact.fixtureB.userData.name : contact.fixtureB}
        keys = list(names.keys())

        # Determine if collision between puck and biscuit
        if any(["puck" in x for x in keys]) and any(["biscuit" in x for x in keys]):
            # Retrieve fixtures
            puck = names[min(keys, key=len)]
            biscuit = names[max(keys, key=len)]
            # Disable contact
            contact.enabled = False
            # Mark biscuit for deletion
            self.collision_list.append((puck, biscuit))
        # Determine if collision between puck and ball
        if any(["puck" in x for x in keys]) and any(["ball" in x for x in keys]):
            # Retrieve fixtures
            puck = names[max(keys, key=len)]
            ball = names[min(keys, key=len)]
            # Do something ...

    def PostSolve(self, contact, impulse):
        # Find out what the collision response was
        pass

# --- create the world ---
world = world(contactListener=myContactListener(), gravity=(0, 0), doSleep=True)

# --- static bodies ---
wall_bottom = world.CreateStaticBody(
    position=(0, 0),
    shapes=edgeShape(vertices=[(0,0), (KG_BOARD_WIDTH * LENGTH_SCALER, 0)])
)
wall_left = world.CreateStaticBody(
    position=(0, 0),
    shapes=edgeShape(vertices=[(0,0), (0, KG_BOARD_HEIGHT * LENGTH_SCALER)])
)
wall_right = world.CreateStaticBody(
    position=(0, 0),
    shapes=edgeShape(vertices=[(KG_BOARD_WIDTH * LENGTH_SCALER, 0), (KG_BOARD_WIDTH * LENGTH_SCALER, KG_BOARD_HEIGHT * LENGTH_SCALER)])
)
wall_top = world.CreateStaticBody(
    position=(0, 0),
    shapes=edgeShape(vertices=[(0, KG_BOARD_HEIGHT * LENGTH_SCALER), (KG_BOARD_WIDTH * LENGTH_SCALER, KG_BOARD_HEIGHT * LENGTH_SCALER)])
)

divider_left = world.CreateStaticBody(
    position=(0, 0),
    shapes=edgeShape(vertices=[(KG_BOARD_WIDTH * LENGTH_SCALER / 2 - KG_DIVIDER_WIDTH * LENGTH_SCALER / 2, 0), (KG_BOARD_WIDTH * LENGTH_SCALER / 2 - KG_DIVIDER_WIDTH * LENGTH_SCALER / 2, KG_BOARD_HEIGHT * LENGTH_SCALER)])
)
divider_left.fixtures[0].filterData.categoryBits=0x0010

divider_right = world.CreateStaticBody(
    position=(0, 0),
    shapes=edgeShape(vertices=[(KG_BOARD_WIDTH * LENGTH_SCALER / 2 + KG_DIVIDER_WIDTH * LENGTH_SCALER / 2, 0), (KG_BOARD_WIDTH * LENGTH_SCALER / 2 + KG_DIVIDER_WIDTH * LENGTH_SCALER / 2, KG_BOARD_HEIGHT * LENGTH_SCALER)])
)
divider_right.fixtures[0].filterData.categoryBits=0x0010

ground = world.CreateStaticBody(position=(0,0))

# --- dynamic bodies ---
@dataclass
class fixtureUserData:
    name: str
    color: tuple

puck1_body = world.CreateDynamicBody(position=(KG_BOARD_WIDTH * LENGTH_SCALER / 3, KG_BOARD_HEIGHT * LENGTH_SCALER / 2), fixedRotation=True, bullet=True)
puck1 = puck1_body.CreateCircleFixture(radius=KG_PUCK_RADIUS * LENGTH_SCALER, restitution=0.0, userData=fixtureUserData("puck1", KG_PUCK_COLOR), density=KG_PUCK_MASS / (pi * (KG_PUCK_RADIUS * LENGTH_SCALER)**2))

puck2_body = world.CreateDynamicBody(position=(2 * KG_BOARD_WIDTH * LENGTH_SCALER / 3, KG_BOARD_HEIGHT * LENGTH_SCALER / 2), fixedRotation=True, bullet=True)
puck2 = puck2_body.CreateCircleFixture(radius=KG_PUCK_RADIUS * LENGTH_SCALER, restitution=0.0, userData=fixtureUserData("puck2", KG_PUCK_COLOR), density=KG_PUCK_MASS / (pi * (KG_PUCK_RADIUS * LENGTH_SCALER)**2))

ball_start_positions = [(KG_BOARD_WIDTH * LENGTH_SCALER - KG_CORNER_RADIUS * LENGTH_SCALER / 2, KG_BOARD_HEIGHT * LENGTH_SCALER - KG_CORNER_RADIUS * LENGTH_SCALER / 2),    # Top right corner
                        (KG_BOARD_WIDTH * LENGTH_SCALER - KG_CORNER_RADIUS * LENGTH_SCALER / 2, KG_CORNER_RADIUS * LENGTH_SCALER / 2),                                      # Bottom right corner
                        (KG_CORNER_RADIUS * LENGTH_SCALER / 2, KG_BOARD_HEIGHT * LENGTH_SCALER - KG_CORNER_RADIUS * LENGTH_SCALER / 2),                                     # Top left corner
                        (KG_CORNER_RADIUS * LENGTH_SCALER / 2, KG_CORNER_RADIUS * LENGTH_SCALER / 2)]                                                                       # Bottom left corner
ball_start_position = choice(ball_start_positions)

ball_body = world.CreateDynamicBody(position=ball_start_position, bullet=True)
ball = ball_body.CreateCircleFixture(radius=KG_BALL_RADIUS * LENGTH_SCALER, restitution=KG_RESTITUTION_COEF, userData=fixtureUserData("ball", KG_BALL_COLOR), density=KG_BALL_MASS / (pi * (KG_BALL_RADIUS * LENGTH_SCALER)**2), maskBits=0xFF0F)

biscuit1_body = world.CreateDynamicBody(position=(KG_BOARD_WIDTH * LENGTH_SCALER / 2, KG_BOARD_HEIGHT * LENGTH_SCALER / 2), bullet=True)
biscuit1 = biscuit1_body.CreateCircleFixture(radius=KG_BISCUIT_RADIUS * LENGTH_SCALER, restitution=KG_RESTITUTION_COEF, userData=fixtureUserData("biscuit1", KG_BISCUIT_COLOR), density=KG_BISCUIT_MASS / (pi * (KG_BISCUIT_RADIUS * LENGTH_SCALER)**2), maskBits=0xFF0F)

biscuit2_body = world.CreateDynamicBody(position=(KG_BOARD_WIDTH * LENGTH_SCALER / 2, (KG_BOARD_HEIGHT * LENGTH_SCALER / 2) + KG_BISCUIT_START_OFFSET_Y * LENGTH_SCALER), bullet=True)
biscuit2 = biscuit2_body.CreateCircleFixture(radius=KG_BISCUIT_RADIUS * LENGTH_SCALER, restitution=KG_RESTITUTION_COEF, userData=fixtureUserData("biscuit2", KG_BISCUIT_COLOR), density=KG_BISCUIT_MASS / (pi * (KG_BISCUIT_RADIUS * LENGTH_SCALER)**2), maskBits=0xFF0F)

biscuit3_body = world.CreateDynamicBody(position=(KG_BOARD_WIDTH * LENGTH_SCALER / 2, (KG_BOARD_HEIGHT * LENGTH_SCALER / 2) - KG_BISCUIT_START_OFFSET_Y * LENGTH_SCALER), bullet=True)
biscuit3 = biscuit3_body.CreateCircleFixture(radius=KG_BISCUIT_RADIUS * LENGTH_SCALER, restitution=KG_RESTITUTION_COEF, userData=fixtureUserData("biscuit3", KG_BISCUIT_COLOR), density=KG_BISCUIT_MASS / (pi * (KG_BISCUIT_RADIUS * LENGTH_SCALER)**2), maskBits=0xFF0F)

biscuit_bodies = [biscuit1_body, biscuit2_body, biscuit3_body]
render_bodies = [puck1_body, puck2_body, ball_body, biscuit1_body, biscuit2_body, biscuit3_body]

# --- create joints ---
ball_ground_joint = world.CreateFrictionJoint(bodyA=ground, bodyB=ball_body, maxForce=ball_body.mass*KG_GRAVITY)
biscuit1_ground_joint = world.CreateFrictionJoint(bodyA=ground, bodyB=biscuit1_body, maxForce=biscuit1_body.mass*KG_GRAVITY)
biscuit2_ground_joint = world.CreateFrictionJoint(bodyA=ground, bodyB=biscuit2_body, maxForce=biscuit2_body.mass*KG_GRAVITY)
biscuit3_ground_joint = world.CreateFrictionJoint(bodyA=ground, bodyB=biscuit3_body, maxForce=biscuit3_body.mass*KG_GRAVITY)
puck1_mouse_joint = None

def apply_magnet_force(puck_body, biscuit_body, permeability, magnetic_charge):
    # Get the distance vector between the two bodies
    force = (puck_body.position - biscuit_body.position)

    # Normalize the distance vector and get the Euclidean distance between the two bodies
    separation = force.Normalize()

    # Compute magnetic force between two points
    force *= (permeability * magnetic_charge**2) / (4 * pi * separation**2)

    # Apply forces to bodies
    biscuit_body.ApplyForceToCenter(force=force, wake=True)

# --- render methods ---
def draw_circle_fixture(circle, pixels_per_meter, surface):
    position = circle.body.transform * circle.shape.pos * pixels_per_meter
    position = (position[0], surface.get_height() - position[1])
    pygame.draw.circle(screen, circle.userData.color, [int(x) for x in position], int(circle.shape.radius * pixels_per_meter))

# --- game state methods ---
def is_body_in_goal(body):
    # return 0 for no, return 1 for left goal, return 2 for right goal

    if dist(body.position, (KG_GOAL_OFFSET_X * LENGTH_SCALER, (KG_BOARD_HEIGHT / 2) * LENGTH_SCALER)) <= KG_GOAL_RADIUS * LENGTH_SCALER:
        return 1
    # if dist(body.position, ((KG_BOARD_WIDTH - KG_GOAL_OFFSET_X) * LENGTH_SCALER, (KG_BOARD_HEIGHT / 2) * LENGTH_SCALER)) <= KG_GOAL_RADIUS * LENGTH_SCALER:
    #     return 2
    
    return 0

def num_biscuits_on_puck(puck_body):
    return len(puck_body.fixtures) - 1

def determine_game_state(puck1_body, puck2_body, ball_body):
    # return 0 for "play" state, 1 for "puck_1 win" state, 2 for "puck_2 win" state

    # Determine puck_1 win conditions
    if is_body_in_goal(puck2_body) or is_body_in_goal(ball_body) == 2 or num_biscuits_on_puck(puck2_body) >= 2:
        return 1

    # Determine puck_2 win conditions
    if is_body_in_goal(puck1_body) or is_body_in_goal(ball_body) == 1 or num_biscuits_on_puck(puck1_body) >= 2:
        return 2

    return 0

# --- main game loop ---
game_board = render_game_board(PPM * LENGTH_SCALER)

running = True
while running:
    # Check the event queue
    for event in pygame.event.get():
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            # The user closed the window or pressed escape
            running = False

        # Control Puck 1
        if event.type == MOUSEBUTTONDOWN:
            if puck1_mouse_joint:
                world.DestroyJoint(puck1_mouse_joint)
            puck1_mouse_joint = world.CreateMouseJoint(bodyA=ground, bodyB=puck1_body, target=puck1_body.position, maxForce=10000)#, dampingRatio=1.0, frequencyHz=60)
        elif event.type == MOUSEBUTTONUP:
            if puck1_mouse_joint:
                world.DestroyJoint(puck1_mouse_joint)
            puck1_mouse_joint = None
            puck1_body.linearVelocity = (0, 0)
        elif event.type == MOUSEMOTION:
            x, y = pygame.mouse.get_pos()
            if puck1_mouse_joint:
                puck1_mouse_joint.target = (x / PPM, (SCREEN_HEIGHT - y) / PPM)

    # Control Puck 2
    puck2_body.linearVelocity = vec2(0, 0)

    # Apply Magnetic Force
    for biscuit_body in biscuit_bodies:
        apply_magnet_force(puck1_body, biscuit_body, KG_PERMEABILITY_AIR, KG_MAGNETIC_CHARGE)
        apply_magnet_force(puck2_body, biscuit_body, KG_PERMEABILITY_AIR, KG_MAGNETIC_CHARGE)

    # Render the world
    screen.blit(game_board, (0,0))

    # Draw the world
    for body in render_bodies:
        for fixture in body:
            draw_circle_fixture(fixture, PPM, screen)

    # Control Puck 1
    # keys = pygame.key.get_pressed()
    # force = 100000
    # # puck1_body.linearVelocity= vec2(0,0)
    # if keys[K_w]:
    #     # puck1_body.ApplyForceToCenter(vec2(0, force), wake=True)
    #     puck1_body.ApplyLinearImpulse(vec2(0, force), puck1_body.position, wake=True)
    # if keys[K_s]:
    #     #  puck1_body.ApplyForceToCenter(vec2(0, -force), wake=True)
    #     puck1_body.ApplyLinearImpulse(vec2(0, -force), puck1_body.position, wake=True)
    # if keys[K_a]:
    #     #  puck1_body.ApplyForceToCenter(vec2(-force, 0), wake=True)
    #     puck1_body.ApplyLinearImpulse(vec2(-force, 0), puck1_body.position, wake=True)
    # if keys[K_d]:
    #     # puck1_body.ApplyForceToCenter(vec2(force, 0), wake=True)
    #     puck1_body.ApplyLinearImpulse(vec2(force, 0), puck1_body.position, wake=True)

    # Make Box2D simulate the physics of our world for one step.
    world.Step(TIME_STEP, 10, 10)

    # Handle collisions
    while world.contactListener.collision_list:
        # Retrieve fixtures
        puck, biscuit = world.contactListener.collision_list.pop()
        
        # Compute new biscuit position
        position = (biscuit.body.position - puck.body.position)
        position.Normalize()
        position = position * (puck.shape.radius + biscuit.shape.radius)

        # Create new biscuit fixture
        new_biscuit = puck.body.CreateCircleFixture(radius=biscuit.shape.radius, pos=position, userData=biscuit.userData)
        new_biscuit.sensor = True

        # Remove old biscuit body
        biscuit_bodies.remove(biscuit.body)
        render_bodies.remove(biscuit.body)
        world.DestroyBody(biscuit.body)

    # Flip the screen and try to keep at the target FPS
    pygame.display.flip()
    clock.tick(TARGET_FPS)
    # print(clock.get_fps())

pygame.quit()
