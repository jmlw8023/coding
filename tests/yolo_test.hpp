#pragma once

#include <yolo.hpp>
#include <cassert>
#include <iostream>
#include <vector>

namespace yolo {
namespace test {

bool testVersionEnum();
bool testBackendEnum();
bool testVersionNames();
bool testBackendNames();
bool testCreateDetector();
bool testDefaultConfig();
bool testDetectResult();
bool testImageDataCreation();
bool testDetectorNotInitialized();
bool testAllBackends();

} // namespace test
} // namespace yolo