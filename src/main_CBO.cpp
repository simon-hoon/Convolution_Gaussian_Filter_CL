//
//  main_CBO.cpp
//
//  Written for CSEG437_CSE5437
//  Department of Computer Science and Engineering
//  Copyright © 2020 Sogang University. All rights reserved.
//
#pragma warning(disable:4996)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

#include <FreeImage/FreeImage.h>

#include "../Util/my_OpenCL_util_2_1.h"
#include "Context_CBO.h"
#include "Config_CBO.h"

Context context_CBO;
cl_int errcode_ret;

FILE* fp_stat;
char tmp_string[512];

__int64 _start, _freq, _end;
float compute_time;
int HFS_HOST;

void read_input_image_from_file32(const char* filename) {
    // Assume everything is fine with reading image from file: no error checking is done.
    context_CBO.image_format = FreeImage_GetFileType(filename, 0);
    FIBITMAP* fi_bitmap = FreeImage_Load(context_CBO.image_format, filename);
    context_CBO.image_width = FreeImage_GetWidth(fi_bitmap);
    context_CBO.image_height = FreeImage_GetHeight(fi_bitmap);
    int bits_per_pixel = FreeImage_GetBPP(fi_bitmap);

    fprintf(stdout, "*** A %d-bit image of %d x %d pixels was read from %s.\n", bits_per_pixel, 
        context_CBO.image_width, context_CBO.image_height, filename);
 
    if (bits_per_pixel == 32) {
        context_CBO.input.fi_bitmap_32 = fi_bitmap;
    }
    else {
        fprintf(stdout, "*** Converting texture from %d bits to 32 bits...\n", bits_per_pixel);
        context_CBO.input.fi_bitmap_32 = FreeImage_ConvertTo32Bits(fi_bitmap);
    }
    FreeImage_Unload(fi_bitmap);

    context_CBO.image_pitch = FreeImage_GetPitch(context_CBO.input.fi_bitmap_32);
    printf("*** input image: width = %d, height = %d, bpp = %d, pitch = %d\n", context_CBO.image_width, 
        context_CBO.image_height, 32, context_CBO.image_pitch);

    context_CBO.input.image_data = FreeImage_GetBits(context_CBO.input.fi_bitmap_32);
}

void prepare_output_image(void) {
    context_CBO.image_data_bytes 
        = context_CBO.image_width * context_CBO.image_height * sizeof(unsigned char) * 4;

    if (context_CBO.image_data_bytes != context_CBO.image_pitch * context_CBO.image_height) {
        fprintf(stderr, "*** Error: do something!\n");
        exit(-1);
    }
    context_CBO.output.image_data = (BYTE *)malloc(sizeof(BYTE) * context_CBO.image_data_bytes);
    if (context_CBO.output.image_data == NULL) {
        fprintf(stderr, "*** Error: cannot allocate memory for output image...\n");
        exit(-1);
    }
}

void write_output_image_to_file32(const char* filename) {
    // Assume everything is fine with writing image into file: no error checking is done.
    context_CBO.output.fi_bitmap_32 = FreeImage_ConvertFromRawBits(context_CBO.output.image_data,
        context_CBO.image_width, context_CBO.image_height, context_CBO.image_pitch, 32,
        FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, FALSE);

    FreeImage_Save(FIF_PNG, context_CBO.output.fi_bitmap_32, filename, 0);
    fprintf(stdout, "*** A 32-bit image of %d x %d pixels was written to %s.\n",
        context_CBO.image_width, context_CBO.image_height, filename);
}

int initialize_OpenCL(void) {
    /* Get the first platform. */
    errcode_ret = clGetPlatformIDs(1, &context_CBO.platform_id, NULL);
    // You may skip error checking if you think it is unnecessary.
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    /* Get the first GPU device. */
    errcode_ret = clGetDeviceIDs(context_CBO.platform_id, CL_DEVICE_TYPE_GPU, 1, &context_CBO.device_id, NULL);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    fprintf(stdout, "\n^^^ The first GPU device on the platform ^^^\n");
    print_device_0(context_CBO.device_id);

    /* Create a context with the devices. */
    context_CBO.context = clCreateContext(NULL, 1, &context_CBO.device_id, NULL, NULL, &errcode_ret);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    /* Create a command-queue for the GPU device. */
    // Use clCreateCommandQueueWithProperties() for OpenCL 2.0.
    context_CBO.cmd_queue = clCreateCommandQueue(context_CBO.context, context_CBO.device_id, 
        CL_QUEUE_PROFILING_ENABLE, &errcode_ret);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    /* Create a program from OpenCL C source code. */
    context_CBO.prog_src.length = read_kernel_from_file(OPENCL_C_PROG_FILE_NAME, 
        &context_CBO.prog_src.string);
    context_CBO.program = clCreateProgramWithSource(context_CBO.context, 1, 
        (const char**)&context_CBO.prog_src.string, &context_CBO.prog_src.length, &errcode_ret);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    /* Build a program executable from the program object. */
    const char options[] = "-cl-std=CL1.2";
    errcode_ret = clBuildProgram(context_CBO.program, 1, &context_CBO.device_id, options, NULL, NULL);
    if (errcode_ret != CL_SUCCESS) {
        print_build_log(context_CBO.program, context_CBO.device_id, "GPU");
        return 1;
    }

    /* Create the kernel from the program. */
    context_CBO.kernel = clCreateKernel(context_CBO.program, KERNEL_NAME, &errcode_ret);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    /* Create input and output buffer objects. */
    context_CBO.BO_input = clCreateBuffer(context_CBO.context, CL_MEM_READ_ONLY,
        sizeof(BYTE) * context_CBO.image_data_bytes, NULL, &errcode_ret);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    context_CBO.BO_output = clCreateBuffer(context_CBO.context, CL_MEM_WRITE_ONLY,
        sizeof(BYTE) * context_CBO.image_data_bytes, NULL, &errcode_ret);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    context_CBO.BO_filter = clCreateBuffer(context_CBO.context, CL_MEM_READ_ONLY,
        sizeof(float) * context_CBO.gaussian_filter.width * context_CBO.gaussian_filter.width, NULL, &errcode_ret);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    fprintf(stdout, "    [Data Transfer to GPU] \n");

    CHECK_TIME_START(_start, _freq);
    // Move the input data from the host memory to the GPU device memory.
    errcode_ret = clEnqueueWriteBuffer(context_CBO.cmd_queue, context_CBO.BO_input, CL_FALSE, 0,
        sizeof(BYTE) * context_CBO.image_data_bytes, context_CBO.input.image_data, 0, NULL, NULL);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    errcode_ret = clEnqueueWriteBuffer(context_CBO.cmd_queue, context_CBO.BO_filter, CL_FALSE, 0,
        sizeof(float) * context_CBO.gaussian_filter.width * context_CBO.gaussian_filter.width, 
        context_CBO.gaussian_filter.weights, 0, NULL, NULL);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    /* Wait until all data transfers finish. */
    clFinish(context_CBO.cmd_queue);  
    CHECK_TIME_END(_start, _end, _freq, compute_time);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    fprintf(stdout, "      * Time by host clock = %.3fms\n\n", compute_time);
    return 0;
}

int set_local_work_size_and_kernel_arguments(void) {
    context_CBO.global_work_size[0] = context_CBO.image_width;
    context_CBO.global_work_size[1] = context_CBO.image_height;

    context_CBO.local_work_size[0] = LOCAL_WORK_SIZE_0;
    context_CBO.local_work_size[1] = LOCAL_WORK_SIZE_1;

    /* Set the kenel arguments. */
#if  KERNEL_SELECTION == 0 || KERNEL_SELECTION == 1 || KERNEL_SELECTION == 15 \
        || KERNEL_SELECTION == 17  || KERNEL_SELECTION == 19 || KERNEL_SELECTION == 111
    errcode_ret = clSetKernelArg(context_CBO.kernel, 0, sizeof(cl_mem), &context_CBO.BO_input);
    errcode_ret |= clSetKernelArg(context_CBO.kernel, 1, sizeof(cl_mem), &context_CBO.BO_output);
    errcode_ret |= clSetKernelArg(context_CBO.kernel, 2, sizeof(int), &context_CBO.image_width);
    errcode_ret |= clSetKernelArg(context_CBO.kernel, 3, sizeof(int), &context_CBO.image_height);
    errcode_ret |= clSetKernelArg(context_CBO.kernel, 4, sizeof(cl_mem), &context_CBO.BO_filter);
    errcode_ret |= clSetKernelArg(context_CBO.kernel, 5, sizeof(int), &context_CBO.gaussian_filter.width);
#elif KERNEL_SELECTION == 2  || KERNEL_SELECTION == 25 || KERNEL_SELECTION == 27  \
        || KERNEL_SELECTION == 29 || KERNEL_SELECTION == 211
    errcode_ret = clSetKernelArg(context_CBO.kernel, 0, sizeof(cl_mem), &context_CBO.BO_input);
    errcode_ret |= clSetKernelArg(context_CBO.kernel, 1, sizeof(cl_mem), &context_CBO.BO_output);
    errcode_ret |= clSetKernelArg(context_CBO.kernel, 2, sizeof(int), &context_CBO.image_width);
    errcode_ret |= clSetKernelArg(context_CBO.kernel, 3, sizeof(int), &context_CBO.image_height);
    errcode_ret |= clSetKernelArg(context_CBO.kernel, 4, sizeof(cl_mem), &context_CBO.BO_filter);
    errcode_ret |= clSetKernelArg(context_CBO.kernel, 5, sizeof(int), &context_CBO.gaussian_filter.width);

    int twice_half_filter_width = 2 * (context_CBO.gaussian_filter.width / 2);
    size_t local_mem_size = sizeof(cl_uchar4)
        * (context_CBO.local_work_size[0] + twice_half_filter_width)
        * (context_CBO.local_work_size[1] + twice_half_filter_width);
    errcode_ret |= clSetKernelArg(context_CBO.kernel, 6, local_mem_size, NULL);
    fprintf(stdout, "^^^ Necessary local memory = %d bytes (%d, %d, %d) ^^^\n\n", local_mem_size,
        sizeof(cl_uchar4), context_CBO.local_work_size[0] + twice_half_filter_width, 
        context_CBO.local_work_size[0] + twice_half_filter_width);
#endif
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    printf_KernelWorkGroupInfo(context_CBO.kernel, context_CBO.device_id);

    return 0;
}

int run_OpenCL_kernel(void) {
    fprintf(stdout, "    [Kernel Execution] \n");

    fp_stat = util_open_stat_file_append(STAT_FILE_NAME);
    util_stamp_stat_file_device_name_and_time(fp_stat, context_CBO.device_id);
    util_reset_event_time();

    CHECK_TIME_START(_start, _freq);
    /* Execute the kernel on the device. */
    for (int i = 0; i < N_EXECUTIONS; i++) {
       errcode_ret = clEnqueueNDRangeKernel(context_CBO.cmd_queue, context_CBO.kernel, 2, NULL, 
           context_CBO.global_work_size, context_CBO.local_work_size, 0, NULL, &context_CBO.event_for_timing);
           if (CHECK_ERROR_CODE(errcode_ret)) return 1;
           clWaitForEvents(1, &context_CBO.event_for_timing);
           if (CHECK_ERROR_CODE(errcode_ret)) return 1;
           util_accumulate_event_times_1_2(context_CBO.event_for_timing);
    }
    CHECK_TIME_END(_start, _end, _freq, compute_time);

    fprintf(stdout, "      * Time by host clock = %.3fms\n\n", compute_time);
    util_print_accumulated_device_time_1_2(N_EXECUTIONS);
    MAKE_STAT_ITEM_LIST_CBO(tmp_string, context_CBO.global_work_size, context_CBO.local_work_size);
    util_stamp_stat_file_ave_device_time_START_to_END_1_2_string(fp_stat, tmp_string);
    util_close_stat_file_append(fp_stat);

    fprintf(stdout, "    [Data Transfer] \n");

    /* Read back the device buffer to the host array. */
    CHECK_TIME_START(_start, _freq);
    errcode_ret = clEnqueueReadBuffer(context_CBO.cmd_queue, context_CBO.BO_output, CL_TRUE, 0,
        sizeof(BYTE) * context_CBO.image_data_bytes, context_CBO.output.image_data, 0, NULL, 
        &context_CBO.event_for_timing);
    CHECK_TIME_END(_start, _end, _freq, compute_time);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    fprintf(stdout, "\n* Time by host clock = %.3fms\n\n", compute_time);
    print_device_time(context_CBO.event_for_timing);

    return 0;
}

void set_Gaussian_filter(int filter_size) {
    context_CBO.gaussian_filter.width = filter_size;

    switch (filter_size) {
    case 5:
        context_CBO.gaussian_filter.weights = Gaussian_5;
        HFS_HOST = 2;
        break;
    case 7:
        context_CBO.gaussian_filter.weights = Gaussian_7;
        HFS_HOST = 3;
        break;
    case 9:
        context_CBO.gaussian_filter.weights = Gaussian_9;
        HFS_HOST = 4;
        break;
    case 11:
        context_CBO.gaussian_filter.weights = Gaussian_11;
        HFS_HOST = 5;
        break;
    default:
        fprintf(stderr, "^^^ Error: invalid Gaussian filter size = %d...\n", filter_size);
        exit(-1);
    }
}

/////////////////////////////////////////////////  HW 01  /////////////////////////////////////////////////

// Process Gaussian filter on CPU

// FS = 5 , HFS = 2
void run_host_5(void) {

    __int64 rows = context_CBO.image_height;
    __int64 cols = context_CBO.image_width;

    fprintf(stdout, "\n    [Host Execution] \n");
    CHECK_TIME_START(_start, _freq);
    for (__int64 i = 0; i < rows; i++) {
        for (__int64 j = 0; j < cols; j++) {

            float sum[4] = { 0.0f, };
            int filter_index = 0;   

            for (__int64 k = -HFS_HOST; k <= HFS_HOST; k++) {

                __int64 r = i + k;
                r = (r < 0) ? 0 : r;
                r = (r >= rows) ? rows - 1 : r;

                { //for (int l = -HFS_HOST; l <= HFS_HOST; l++) {

                    __int64 c = j - 2; // -2
                    c = (c < 0) ? 0 : c;
                    sum[0] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 0) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[1] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 1) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[2] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 2) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[3] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 3) * context_CBO.gaussian_filter.weights[filter_index++];

                    c++; // -1
                    c = (c < 0) ? 0 : c;
                    sum[0] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 0) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[1] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 1) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[2] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 2) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[3] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 3) * context_CBO.gaussian_filter.weights[filter_index++];

                    c++; // 0
                    sum[0] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 0) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[1] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 1) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[2] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 2) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[3] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 3) * context_CBO.gaussian_filter.weights[filter_index++];

                    c++; // 1
                    c = (c >= cols) ? cols - 1 : c;
                    sum[0] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 0) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[1] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 1) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[2] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 2) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[3] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 3) * context_CBO.gaussian_filter.weights[filter_index++];

                    c++; // 2
                    c = (c >= cols) ? cols - 1 : c;
                    sum[0] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 0) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[1] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 1) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[2] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 2) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[3] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 3) * context_CBO.gaussian_filter.weights[filter_index++];
                }
            }
            *(context_CBO.output.image_data + i * cols * 4 + j * 4 + 0) = (BYTE)sum[0];
            *(context_CBO.output.image_data + i * cols * 4 + j * 4 + 1) = (BYTE)sum[1];
            *(context_CBO.output.image_data + i * cols * 4 + j * 4 + 2) = (BYTE)sum[2];
            *(context_CBO.output.image_data + i * cols * 4 + j * 4 + 3) = (BYTE)sum[3];
        }
    }
    CHECK_TIME_END(_start, _end, _freq, compute_time);

    fprintf(stdout, "      * Time by host clock = %.3fms\n\n", compute_time);
}

// FS = 7 , HFS = 3
void run_host_7(void) {

    __int64 rows = context_CBO.image_height;
    __int64 cols = context_CBO.image_width;

    fprintf(stdout, "\n    [Host Execution] \n");
    CHECK_TIME_START(_start, _freq);
    for (__int64 i = 0; i < rows; i++) {
        for (__int64 j = 0; j < cols; j++) {

            float sum[4] = { 0.0f, };
            int filter_index = 0;

            for (__int64 k = -HFS_HOST; k <= HFS_HOST; k++) {

                __int64 r = i + k;
                r = (r < 0) ? 0 : r;
                r = (r >= rows) ? rows - 1 : r;

                { //for (int l = -HFS_HOST; l <= HFS_HOST; l++) {

                    __int64 c = j - 3; // -3
                    c = (c < 0) ? 0 : c;
                    sum[0] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 0) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[1] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 1) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[2] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 2) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[3] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 3) * context_CBO.gaussian_filter.weights[filter_index++];

                    c++; // -2
                    c = (c < 0) ? 0 : c;
                    sum[0] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 0) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[1] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 1) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[2] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 2) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[3] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 3) * context_CBO.gaussian_filter.weights[filter_index++];

                    c++; // -1
                    c = (c < 0) ? 0 : c;
                    sum[0] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 0) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[1] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 1) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[2] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 2) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[3] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 3) * context_CBO.gaussian_filter.weights[filter_index++];

                    c++; // 0
                    sum[0] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 0) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[1] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 1) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[2] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 2) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[3] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 3) * context_CBO.gaussian_filter.weights[filter_index++];

                    c++; // 1
                    c = (c >= cols) ? cols - 1 : c;
                    sum[0] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 0) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[1] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 1) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[2] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 2) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[3] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 3) * context_CBO.gaussian_filter.weights[filter_index++];

                    c++; // 2
                    c = (c >= cols) ? cols - 1 : c;
                    sum[0] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 0) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[1] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 1) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[2] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 2) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[3] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 3) * context_CBO.gaussian_filter.weights[filter_index++];

                    c++; // 3
                    c = (c >= cols) ? cols - 1 : c;
                    sum[0] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 0) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[1] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 1) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[2] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 2) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[3] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 3) * context_CBO.gaussian_filter.weights[filter_index++];
                }
            }
            *(context_CBO.output.image_data + i * cols * 4 + j * 4 + 0) = (BYTE)sum[0];
            *(context_CBO.output.image_data + i * cols * 4 + j * 4 + 1) = (BYTE)sum[1];
            *(context_CBO.output.image_data + i * cols * 4 + j * 4 + 2) = (BYTE)sum[2];
            *(context_CBO.output.image_data + i * cols * 4 + j * 4 + 3) = (BYTE)sum[3];
        }
    }
    CHECK_TIME_END(_start, _end, _freq, compute_time);

    fprintf(stdout, "      * Time by host clock = %.3fms\n\n", compute_time);
}

// FS = 9 , HFS = 4
void run_host_9(void) {

    __int64 rows = context_CBO.image_height;
    __int64 cols = context_CBO.image_width;
  
    fprintf(stdout, "\n    [Host Execution] \n");
    CHECK_TIME_START(_start, _freq);
    for (__int64 i = 0; i < rows; i++) {
        for (__int64 j = 0; j < cols; j++) {

            float sum[4] = { 0.0f, };
            int filter_index = 0;

            for (__int64 k = -HFS_HOST; k <= HFS_HOST; k++) {

                __int64 r = i + k;
                r = (r < 0) ? 0 : r;
                r = (r >= rows) ? rows - 1 : r;

                { //for (int l = -HFS_HOST; l <= HFS_HOST; l++) {

                    __int64 c = j - 4; // -4
                    c = (c < 0) ? 0 : c;
                    sum[0] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 0) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[1] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 1) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[2] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 2) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[3] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 3) * context_CBO.gaussian_filter.weights[filter_index++];

                    c++; // -3
                    c = (c < 0) ? 0 : c;
                    sum[0] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 0) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[1] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 1) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[2] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 2) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[3] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 3) * context_CBO.gaussian_filter.weights[filter_index++];

                    c++; // -2
                    c = (c < 0) ? 0 : c;
                    sum[0] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 0) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[1] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 1) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[2] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 2) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[3] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 3) * context_CBO.gaussian_filter.weights[filter_index++];

                    c++; // -1
                    c = (c < 0) ? 0 : c;
                    sum[0] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 0) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[1] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 1) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[2] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 2) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[3] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 3) * context_CBO.gaussian_filter.weights[filter_index++];

                    c++; // 0
                    sum[0] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 0) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[1] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 1) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[2] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 2) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[3] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 3) * context_CBO.gaussian_filter.weights[filter_index++];

                    c++; // 1
                    c = (c >= cols) ? cols - 1 : c;
                    sum[0] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 0) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[1] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 1) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[2] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 2) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[3] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 3) * context_CBO.gaussian_filter.weights[filter_index++];

                    c++; // 2
                    c = (c >= cols) ? cols - 1 : c;
                    sum[0] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 0) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[1] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 1) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[2] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 2) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[3] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 3) * context_CBO.gaussian_filter.weights[filter_index++];

                    c++; // 3
                    c = (c >= cols) ? cols - 1 : c;
                    sum[0] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 0) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[1] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 1) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[2] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 2) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[3] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 3) * context_CBO.gaussian_filter.weights[filter_index++];

                    c++; // 4
                    c = (c >= cols) ? cols - 1 : c;
                    sum[0] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 0) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[1] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 1) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[2] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 2) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[3] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 3) * context_CBO.gaussian_filter.weights[filter_index++];
                }
            }
            *(context_CBO.output.image_data + i * cols * 4 + j * 4 + 0) = (BYTE)sum[0];
            *(context_CBO.output.image_data + i * cols * 4 + j * 4 + 1) = (BYTE)sum[1];
            *(context_CBO.output.image_data + i * cols * 4 + j * 4 + 2) = (BYTE)sum[2];
            *(context_CBO.output.image_data + i * cols * 4 + j * 4 + 3) = (BYTE)sum[3];
        }
    }
    CHECK_TIME_END(_start, _end, _freq, compute_time);

    fprintf(stdout, "      * Time by host clock = %.3fms\n\n", compute_time);
}

// FS = 11 , HFS = 5
void run_host_11(void) {

    __int64 rows = context_CBO.image_height;
    __int64 cols = context_CBO.image_width;

    fprintf(stdout, "\n    [Host Execution] \n");
    CHECK_TIME_START(_start, _freq);
    for (__int64 i = 0; i < rows; i++) {
        for (__int64 j = 0; j < cols; j++) {

            float sum[4] = { 0.0f, };
            int filter_index = 0;

            for (__int64 k = -HFS_HOST; k <= HFS_HOST; k++) {

                __int64 r = i + k;
                r = (r < 0) ? 0 : r;
                r = (r >= rows) ? rows - 1 : r;

                { //for (int l = -HFS_HOST; l <= HFS_HOST; l++) {

                    __int64 c = j - 5; // -5
                    c = (c < 0) ? 0 : c;
                    sum[0] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 0) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[1] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 1) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[2] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 2) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[3] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 3) * context_CBO.gaussian_filter.weights[filter_index++];

                    c++; // -4
                    c = (c < 0) ? 0 : c;
                    sum[0] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 0) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[1] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 1) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[2] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 2) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[3] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 3) * context_CBO.gaussian_filter.weights[filter_index++];

                    c++; // -3
                    c = (c < 0) ? 0 : c;
                    sum[0] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 0) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[1] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 1) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[2] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 2) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[3] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 3) * context_CBO.gaussian_filter.weights[filter_index++];

                    c++; // -2
                    c = (c < 0) ? 0 : c;
                    sum[0] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 0) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[1] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 1) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[2] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 2) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[3] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 3) * context_CBO.gaussian_filter.weights[filter_index++];

                    c++; // -1
                    c = (c < 0) ? 0 : c;
                    sum[0] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 0) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[1] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 1) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[2] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 2) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[3] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 3) * context_CBO.gaussian_filter.weights[filter_index++];

                    c++; // 0
                    sum[0] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 0) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[1] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 1) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[2] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 2) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[3] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 3) * context_CBO.gaussian_filter.weights[filter_index++];

                    c++; // 1
                    c = (c >= cols) ? cols - 1 : c;
                    sum[0] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 0) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[1] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 1) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[2] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 2) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[3] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 3) * context_CBO.gaussian_filter.weights[filter_index++];

                    c++; // 2
                    c = (c >= cols) ? cols - 1 : c;
                    sum[0] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 0) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[1] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 1) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[2] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 2) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[3] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 3) * context_CBO.gaussian_filter.weights[filter_index++];

                    c++; // 3
                    c = (c >= cols) ? cols - 1 : c;
                    sum[0] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 0) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[1] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 1) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[2] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 2) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[3] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 3) * context_CBO.gaussian_filter.weights[filter_index++];

                    c++; // 4
                    c = (c >= cols) ? cols - 1 : c;
                    sum[0] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 0) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[1] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 1) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[2] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 2) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[3] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 3) * context_CBO.gaussian_filter.weights[filter_index++];

                    c++; // 5
                    c = (c >= cols) ? cols - 1 : c;
                    sum[0] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 0) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[1] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 1) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[2] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 2) * context_CBO.gaussian_filter.weights[filter_index];
                    sum[3] += *(context_CBO.input.image_data + r * cols * 4 + c * 4 + 3) * context_CBO.gaussian_filter.weights[filter_index++];
                }
            }
            *(context_CBO.output.image_data + i * cols * 4 + j * 4 + 0) = (BYTE)sum[0];
            *(context_CBO.output.image_data + i * cols * 4 + j * 4 + 1) = (BYTE)sum[1];
            *(context_CBO.output.image_data + i * cols * 4 + j * 4 + 2) = (BYTE)sum[2];
            *(context_CBO.output.image_data + i * cols * 4 + j * 4 + 3) = (BYTE)sum[3];
        }
    }
    CHECK_TIME_END(_start, _end, _freq, compute_time);

    fprintf(stdout, "      * Time by host clock = %.3fms\n\n", compute_time);
}

/////////////////////////////////////////////////  HW 01 /////////////////////////////////////////////////
 
void clean_up_system(void) {
     // Free OpenCL and other resources. 
    if (context_CBO.prog_src.string) free(context_CBO.prog_src.string);
    if (context_CBO.BO_input) clReleaseMemObject(context_CBO.BO_input);
    if (context_CBO.BO_input) clReleaseMemObject(context_CBO.BO_output);
    if (context_CBO.BO_input) clReleaseMemObject(context_CBO.BO_filter);
    if (context_CBO.kernel) clReleaseKernel(context_CBO.kernel);
    if (context_CBO.program) clReleaseProgram(context_CBO.program);
    if (context_CBO.cmd_queue) clReleaseCommandQueue(context_CBO.cmd_queue);
    if (context_CBO.device_id) clReleaseDevice(context_CBO.device_id);
    if (context_CBO.context) clReleaseContext(context_CBO.context);
    if (context_CBO.event_for_timing) clReleaseEvent(context_CBO.event_for_timing);
    // what else?
};

int main(int argc, char* argv[]) {

#if KERNEL_SELECTION != 3
    char program_name[] = "Sogang CSEG475_5475 Image Processing/Convolution_Buffer_Object(BO)";
    fprintf(stdout, "\n###  %s\n\n", program_name);
    fprintf(stdout, "/////////////////////////////////////////////////////////////////////////\n");
    fprintf(stdout, "### INPUT FILE NAME = \t\t%s\n", INPUT_FILE_NAME);
    fprintf(stdout, "### OUTPUT FILE NAME = \t\t%s\n", OUTPUT_FILE_NAME);
    fprintf(stdout, "### GAUSSIAN FILTER SIZE = \t%d\n\n", GAUSSIAN_FILTER_SIZE);

    fprintf(stdout, "### NUMBER OF EXECUTIONS = \t%d\n\n", N_EXECUTIONS);

    fprintf(stdout, "### WORK-GROUP SIZE = \t\t(%d, %d)\n", LOCAL_WORK_SIZE_0, LOCAL_WORK_SIZE_1);
    fprintf(stdout, "### KERNEL SELECTION = \t\t%d\n", KERNEL_SELECTION);
    fprintf(stdout, "/////////////////////////////////////////////////////////////////////////\n\n");

    read_input_image_from_file32(INPUT_FILE_NAME);
    prepare_output_image();
    set_Gaussian_filter(GAUSSIAN_FILTER_SIZE);

    int flag = initialize_OpenCL();
    if (flag) goto finish;
    flag = set_local_work_size_and_kernel_arguments();
    if (flag) goto finish;
    flag = run_OpenCL_kernel();
    if (flag) goto finish;

    write_output_image_to_file32(OUTPUT_FILE_NAME);
finish:
    clean_up_system();
    return flag;

#else
    char program_name[] = "Sogang CSEG475_5475 HW_01";
    fprintf(stdout, "\n###  %s\n\n", program_name);
    fprintf(stdout, "/////////////////////////////////////////////////////////////////////////\n");
    fprintf(stdout, "### INPUT FILE NAME = \t\t%s\n", INPUT_FILE_NAME);
    fprintf(stdout, "### OUTPUT FILE NAME = \t\t%s\n", OUTPUT_FILE_NAME);
    fprintf(stdout, "### GAUSSIAN FILTER SIZE = \t%d\n\n", GAUSSIAN_FILTER_SIZE);

    fprintf(stdout, "### KERNEL SELECTION = \t\t%d\n", KERNEL_SELECTION);
    fprintf(stdout, "  => THIS IS HOST EXECUTION \t*** KERNEL IS NOT AVAILABLE *** \n\n");
    fprintf(stdout, "/////////////////////////////////////////////////////////////////////////\n\n");

    read_input_image_from_file32(INPUT_FILE_NAME);
    prepare_output_image();
    set_Gaussian_filter(GAUSSIAN_FILTER_SIZE);
    
    switch (GAUSSIAN_FILTER_SIZE) { // run on CPU for each FILTER SIZE
    case 5: run_host_5(); break;
    case 7: run_host_7(); break;
    case 9: run_host_9(); break;
    case 11: run_host_11(); break;
    }
   
    write_output_image_to_file32(OUTPUT_FILE_NAME_HOST); // store image data processed on host program
#endif
}

