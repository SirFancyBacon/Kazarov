#include <SDL3/SDL.h>
#include <gtest/gtest.h>

#include "platform/input.hpp"

TEST(KeyboardWalk, WalkBindingHalvesKeyboardMovementOutsideMenus) {
    InputResetBindings();
    InputSetMenuNavigation(false);
    InputKeyboardMouse held;
    held.keys = {SDL_SCANCODE_W};
    const auto run = InputApplyKeyboardMouse({}, held);
    EXPECT_EQ(run.left_y, InputStickByte(-1.0f));

    held.keys.push_back(SDL_SCANCODE_LSHIFT);
    const auto walk = InputApplyKeyboardMouse({}, held);
    EXPECT_EQ(walk.left_y, InputStickByte(-0.5f));
    EXPECT_EQ(walk.left_x, kInputAxisCentre);
    // The right stick keeps full deflection.
    held.keys.push_back(SDL_SCANCODE_L);
    EXPECT_EQ(InputApplyKeyboardMouse({}, held).right_x, InputStickByte(1.0f));

    // Menus read the stick through their own threshold and keep full deflection.
    InputSetMenuNavigation(true);
    EXPECT_EQ(InputApplyKeyboardMouse({}, held).left_y, InputStickByte(-1.0f));
    InputSetMenuNavigation(false);
}

TEST(KeyboardWalk, ReboundWalkKeyAndControllerSticksAreUnchanged) {
    InputResetBindings();
    const std::string_view binding[] = {"Left Ctrl"};
    ASSERT_TRUE(InputBindKeys("walk", binding));
    InputKeyboardMouse held;
    held.keys = {SDL_SCANCODE_D, SDL_SCANCODE_LCTRL};
    EXPECT_EQ(InputApplyKeyboardMouse({}, held).left_x, InputStickByte(0.5f));

    InputPadState controller;
    controller.left_x = 255;
    EXPECT_EQ(InputApplyKeyboardMouse(controller, held).left_x, 255);
    InputResetBindings();
}
