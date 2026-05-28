/**
 * @file test_main.cc
 * @brief YoloInterface test program entry point
 */

#include "yolo_test.hpp"

int main() {
    std::cout << "==========================================\n";
    std::cout << "       YoloInterface Test Suite          \n";
    std::cout << "==========================================\n\n";

    bool all_passed = true;

    all_passed &= yolo::test::testVersionEnum();
    all_passed &= yolo::test::testBackendEnum();
    all_passed &= yolo::test::testVersionNames();
    all_passed &= yolo::test::testBackendNames();
    all_passed &= yolo::test::testCreateDetector();
    all_passed &= yolo::test::testDefaultConfig();
    all_passed &= yolo::test::testDetectResult();
    all_passed &= yolo::test::testImageDataCreation();
    all_passed &= yolo::test::testDetectorNotInitialized();
    all_passed &= yolo::test::testAllBackends();

    std::cout << "\n==========================================\n";
    if (all_passed) {
        std::cout << "       ALL TESTS PASSED                  \n";
    } else {
        std::cout << "       SOME TESTS FAILED                 \n";
    }
    std::cout << "==========================================\n";

    return all_passed ? 0 : 1;
}