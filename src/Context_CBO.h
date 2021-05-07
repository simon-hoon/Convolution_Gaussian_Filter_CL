﻿//
//  Context_CBO.h
//
//  Written for CSEG437_CSE5437
//  Department of Computer Science and Engineering
//  Copyright © 2020 Sogang University. All rights reserved.
//

#ifndef __OPENCL_STUFFS_H__
#define __OPENCL_STUFFS_H__

#include <CL/cl.h>
#include <CL/cl_gl.h>

#define     QUEUED_TO_END         0
#define     SUBMIT_TO_END          1
#define     START_TO_END            2

typedef struct _OPENCL_C_PROG_SRC {
    size_t length;
    char* string;
} OPENCL_C_PROG_SRC;

typedef struct _Context {
    FREE_IMAGE_FORMAT image_format;
    unsigned int image_width, image_height, image_pitch;
    size_t image_data_bytes;

    struct {
        int width;
        const float* weights;
    } gaussian_filter;

    struct {
        FIBITMAP* fi_bitmap_32;
        BYTE* image_data;
    } input;
    struct {
        FIBITMAP* fi_bitmap_32;
        BYTE* image_data;
    } output;

    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_context context;
    cl_command_queue cmd_queue;
    cl_program program;
    cl_kernel kernel;
    OPENCL_C_PROG_SRC prog_src;
    cl_mem BO_input, BO_output;
    cl_mem BO_filter;
    cl_event event_for_timing;

    cl_uint work_dim;
    size_t global_work_offset[3], global_work_size[3], local_work_size[3];
} Context;

static const float Gaussian_5[25] = {
    0.0037650,  0.0150190,  0.0237920,  0.0150190,  0.0037650,
    0.0150190,  0.0599121,  0.0949072,  0.0599121,  0.0150190,
    0.0237920,  0.0949072,  0.1503423,  0.0949072,  0.0237920,
    0.0150190,  0.0599121,  0.0949072,  0.0599121,  0.0150190,
    0.0037650,  0.0150190,  0.0237920,  0.0150190,  0.0037650
};

static const float Gaussian_7[49] = {
    0.0000360,  0.0003630,  0.0014460,  0.0022910,  0.0014460,  0.0003630,  0.0000360,
    0.0003630,  0.0036760,  0.0146619,  0.0232258,  0.0146619,  0.0036760,  0.0003630,
    0.0014460,  0.0146619,  0.0584875,  0.0926503,  0.0584875,  0.0146619,  0.0014460,
    0.0022910,  0.0232258,  0.0926503,  0.1467668,  0.0926503,  0.0232258,  0.0022910,
    0.0014460,  0.0146619,  0.0584875,  0.0926503,  0.0584875,  0.0146619,  0.0014460,
    0.0003630,  0.0036760,  0.0146619,  0.0232258,  0.0146619,  0.0036760,  0.0003630,
    0.0000360,  0.0003630,  0.0014460,  0.0022910,  0.0014460,  0.0003630,  0.0000360
};

static const float Gaussian_9[81] = {
    0.0000000,  0.0000010,  0.0000140,  0.0000550,  0.0000880,  0.0000550,  0.0000140,  0.0000010,  0.0000000,
    0.0000010,  0.0000360,  0.0003620,  0.0014450,  0.0022890,  0.0014450,  0.0003620,  0.0000360,  0.0000010,
    0.0000140,  0.0003620,  0.0036720,  0.0146481,  0.0232051,  0.0146481,  0.0036720,  0.0003620,  0.0000140,
    0.0000550,  0.0014450,  0.0146481,  0.0584343,  0.0925666,  0.0584343,  0.0146481,  0.0014450,  0.0000550,
    0.0000880,  0.0022890,  0.0232051,  0.0925666,  0.1466349,  0.0925666,  0.0232051,  0.0022890,  0.0000880,
    0.0000550,  0.0014450,  0.0146481,  0.0584343,  0.0925666,  0.0584343,  0.0146481,  0.0014450,  0.0000550,
    0.0000140,  0.0003620,  0.0036720,  0.0146481,  0.0232051,  0.0146481,  0.0036720,  0.0003620,  0.0000140,
    0.0000010,  0.0000360,  0.0003620,  0.0014450,  0.0022890,  0.0014450,  0.0003620,  0.0000360,  0.0000010,
    0.0000000,  0.0000010,  0.0000140,  0.0000550,  0.0000880,  0.0000550,  0.0000140,  0.0000010,  0.0000000
};

static const float Gaussian_11[121] = { 
    0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000010, 0.0000010, 0.0000010, 0.0000000, 0.0000000, 0.0000000, 0.0000000,
    0.0000000, 0.0000000, 0.0000010, 0.0000140, 0.0000550, 0.0000880, 0.0000550, 0.0000140, 0.0000010, 0.0000000, 0.0000000,
    0.0000000, 0.0000010, 0.0000360, 0.0003620, 0.0014450, 0.0022890, 0.0014450, 0.0003620, 0.0000360, 0.0000010, 0.0000000,
    0.0000000, 0.0000140, 0.0003620, 0.0036720, 0.0146482, 0.0232043, 0.0146482, 0.0036720, 0.0003620, 0.0000140, 0.0000000,
    0.0000010, 0.0000550, 0.0014450, 0.0146482, 0.0584337, 0.0925651, 0.0584337, 0.0146482, 0.0014450, 0.0000550, 0.0000010,
    0.0000010, 0.0000880, 0.0022890, 0.0232043, 0.0925651, 0.1466338, 0.0925651, 0.0232043, 0.0022890, 0.0000880, 0.0000010,
    0.0000010, 0.0000550, 0.0014450, 0.0146482, 0.0584337, 0.0925651, 0.0584337, 0.0146482, 0.0014450, 0.0000550, 0.0000010,
    0.0000000, 0.0000140, 0.0003620, 0.0036720, 0.0146482, 0.0232043, 0.0146482, 0.0036720, 0.0003620, 0.0000140, 0.0000000,
    0.0000000, 0.0000010, 0.0000360, 0.0003620, 0.0014450, 0.0022890, 0.0014450, 0.0003620, 0.0000360, 0.0000010, 0.0000000,
    0.0000000, 0.0000000, 0.0000010, 0.0000140, 0.0000550, 0.0000880, 0.0000550, 0.0000140, 0.0000010, 0.0000000, 0.0000000,
    0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000010, 0.0000010, 0.0000010, 0.0000000, 0.0000000, 0.0000000, 0.0000000
};

#endif // __OPENCL_STUFFS_H__