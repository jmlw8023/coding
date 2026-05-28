#pragma once

#if defined(YOLO_EXPORTS)
    #define YOLO_API __declspec(dllexport)
#elif defined(YOLO_IMPORTS)
    #define YOLO_API __declspec(dllimport)
#else
    #define YOLO_API
#endif