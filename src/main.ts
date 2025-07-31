import { mat4, vec3 } from "gl-matrix";

async function initWebGPU() {
  const adapter = await navigator.gpu?.requestAdapter();
  const device = await adapter?.requestDevice();

  if (!device) {
    alert("Your browser does not support WebGPU.");
    return;
  }

  const canvas = document.getElementById("webgpu-canvas") as HTMLCanvasElement;

  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;

  const context = canvas.getContext("webgpu") as GPUCanvasContext;

  const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device,
    format: presentationFormat,
    alphaMode: "premultiplied",
    size: [canvas.width, canvas.height],
  });

  let depthTexture = device.createTexture({
    size: [canvas.width, canvas.height, 1],
    format: "depth24plus",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  const cubeVertices = new Float32Array([
    -0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5, 0.5, -0.5, -0.5,
    -0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5, -0.5, -0.5, -0.5, 0.5, -0.5,
    -0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5,
    -0.5, 0.5, -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, 0.5,
    0.5, 0.5, 0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, 0.5, 0.5,
    -0.5, 0.5, -0.5,
  ]);

  const cubeTriangleIndices = new Uint16Array([
    0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7, 8, 9, 10, 8, 10, 11, 12, 13, 14, 12, 14,
    15, 16, 17, 18, 16, 18, 19, 20, 21, 22, 20, 22, 23,
  ]);

  const cubeWireframeVertexPositions = new Float32Array([
    -0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5, 0.5, -0.5, -0.5,
    -0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5, -0.5, -0.5,
  ]);

  const cubeWireframeIndices = new Uint16Array([
    0, 1, 1, 2, 2, 3, 3, 0, 4, 7, 7, 6, 6, 5, 5, 4, 0, 4, 1, 7, 2, 6, 3, 5,
  ]);

  const triangleVertexBuffer = device.createBuffer({
    size: cubeVertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Float32Array(triangleVertexBuffer.getMappedRange()).set(cubeVertices);
  triangleVertexBuffer.unmap();

  const triangleIndexBuffer = device.createBuffer({
    size: cubeTriangleIndices.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint16Array(triangleIndexBuffer.getMappedRange()).set(
    cubeTriangleIndices,
  );
  triangleIndexBuffer.unmap();

  const wireframeVertexBuffer = device.createBuffer({
    size: cubeWireframeVertexPositions.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Float32Array(wireframeVertexBuffer.getMappedRange()).set(
    cubeWireframeVertexPositions,
  );
  wireframeVertexBuffer.unmap();

  const wireframeIndexBuffer = device.createBuffer({
    size: cubeWireframeIndices.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint16Array(wireframeIndexBuffer.getMappedRange()).set(
    cubeWireframeIndices,
  );
  wireframeIndexBuffer.unmap();

  const baseSegmentMatrices: mat4[] = [];

  const SEGMENT_WIDTH = 20.0;
  const FRACTAL_SUBDIVISION_FACTOR = 3.0;
  const MAX_INNER_FRACTAL_LEVELS = 3;

  function generateBaseTunnelFrame(
    level: number,
    currentScale: number,
    parentMatrix: mat4,
  ) {
    if (level >= MAX_INNER_FRACTAL_LEVELS) {
      return;
    }

    const nextScale = currentScale / FRACTAL_SUBDIVISION_FACTOR;
    const offsetDistance = currentScale / 2.0;

    for (let x = -1; x <= 1; x++) {
      for (let y = -1; y <= 1; y++) {
        if (x === 0 && y === 0) {
          continue;
        }

        let childMatrix = mat4.create();
        const translation = vec3.fromValues(
          x * offsetDistance,
          y * offsetDistance,
          0,
        );

        mat4.translate(childMatrix, parentMatrix, translation);
        mat4.scale(
          childMatrix,
          childMatrix,
          vec3.fromValues(nextScale, nextScale, nextScale),
        );

        baseSegmentMatrices.push(childMatrix);

        generateBaseTunnelFrame(level + 1, nextScale, childMatrix);
      }
    }
  }

  generateBaseTunnelFrame(0, 1.0, mat4.create());

  const numCubesPerSegment = baseSegmentMatrices.length;
  console.log(`Base segment generated with ${numCubesPerSegment} cubes.`);

  const TUNNEL_SEGMENT_LENGTH = 20.0;
  const NUM_RENDERING_SEGMENTS = 20; // Increased to ensure continuity
  const TOTAL_RENDERED_LENGTH = NUM_RENDERING_SEGMENTS * TUNNEL_SEGMENT_LENGTH;
  const TOTAL_INSTANCES_IN_BUFFER = NUM_RENDERING_SEGMENTS * numCubesPerSegment;

  const instanceBuffer = device.createBuffer({
    size: TOTAL_INSTANCES_IN_BUFFER * 16 * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    mappedAtCreation: false,
  });

  const currentRenderMatrices = new Float32Array(
    TOTAL_INSTANCES_IN_BUFFER * 16,
  );

  const uniformBufferSize = 80;
  const uniformBuffer = device.createBuffer({
    size: uniformBufferSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const shaderModule = device.createShaderModule({
    code: `
          const PI: f32 = 3.141592653589793;

          struct SceneUniforms {
            viewProjectionMatrix : mat4x4<f32>,
            time : f32,
          };
          @group(0) @binding(0) var<uniform> scene: SceneUniforms;

          struct InstanceInput {
            @location(1) modelMatrixCol0 : vec4<f32>,
            @location(2) modelMatrixCol1 : vec4<f32>,
            @location(3) modelMatrixCol2 : vec4<f32>,
            @location(4) modelMatrixCol3 : vec4<f32>,
          };

          struct VertexOutput {
            @builtin(position) position : vec4<f32>,
            @location(0) worldZ : f32,
          };

          @vertex
          fn vs_main(
            @location(0) position : vec3<f32>,
            instance : InstanceInput
          ) -> VertexOutput {
            var output : VertexOutput;
            let modelMatrix = mat4x4<f32>(
                instance.modelMatrixCol0,
                instance.modelMatrixCol1,
                instance.modelMatrixCol2,
                instance.modelMatrixCol3
            );

            var worldPos = modelMatrix * vec4<f32>(position, 1.0);

            let waveFrequency = 0.05;
            let waveAmplitude = 1.0;
            let displacementX = sin(worldPos.z * waveFrequency + scene.time * 2.0) * waveAmplitude;
            let displacementY = cos(worldPos.z * waveFrequency + scene.time * 2.0) * waveAmplitude;
            worldPos.x += displacementX;
            worldPos.y += displacementY;

            let rotationSpeed = 0.5;
            let rotationAmplitude = 0.5;
            let angle = sin(worldPos.z * 0.1 + scene.time * rotationSpeed) * rotationAmplitude;

            let cos_angle = cos(angle);
            let sin_angle = sin(angle);

            let rotX_mat = mat4x4<f32>(
                vec4<f32>(1.0, 0.0, 0.0, 0.0),
                vec4<f32>(0.0, cos_angle, sin_angle, 0.0),
                vec4<f32>(0.0, -sin_angle, cos_angle, 0.0),
                vec4<f32>(0.0, 0.0, 0.0, 1.0)
            );
            let rotY_mat = mat4x4<f32>(
                vec4<f32>(cos_angle, 0.0, -sin_angle, 0.0),
                vec4<f32>(0.0, 1.0, 0.0, 0.0),
                vec4<f32>(sin_angle, 0.0, cos_angle, 0.0),
                vec4<f32>(0.0, 0.0, 0.0, 1.0)
            );
            worldPos = rotX_mat * worldPos;
            worldPos = rotY_mat * worldPos;

            let zDistance = abs(worldPos.z);
            let scaleFactor = 1.0 + sin(zDistance * 0.1 + scene.time * 1.5) * 0.2;
            worldPos = vec4<f32>(worldPos.xyz * scaleFactor, worldPos.w);

            output.position = scene.viewProjectionMatrix * worldPos;
            output.worldZ = worldPos.z;
            return output;
          }

          @fragment
          fn fs_solid() -> @location(0) vec4<f32> {
            return vec4<f32>(0.0, 0.0, 0.0, 1.0);
          }

          @fragment
          fn fs_wireframe(@location(0) worldZ : f32) -> @location(0) vec4<f32> {
            let brightness = 0.8;
            let animatedColor = vec3<f32>(
                sin(worldZ * 0.1 + scene.time * 1.0) * 0.5 + 0.5,
                cos(worldZ * 0.1 + scene.time * 1.0) * 0.5 + 0.5,
                sin(worldZ * 0.1 + scene.time * 1.0 + PI * 0.5) * 0.5 + 0.5
            ) * brightness;

            return vec4<f32>(animatedColor, 1.0);
          }
        `,
  });

  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [
      device.createBindGroupLayout({
        entries: [
          {
            binding: 0,
            visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
            buffer: { type: "uniform", hasDynamicOffset: false },
          },
        ],
      }),
    ],
  });

  const instanceBufferLayout: GPUVertexBufferLayout = {
    arrayStride: 16 * Float32Array.BYTES_PER_ELEMENT,
    stepMode: "instance",
    attributes: [
      { shaderLocation: 1, offset: 0, format: "float32x4" },
      {
        shaderLocation: 2,
        offset: 4 * Float32Array.BYTES_PER_ELEMENT,
        format: "float32x4",
      },
      {
        shaderLocation: 3,
        offset: 8 * Float32Array.BYTES_PER_ELEMENT,
        format: "float32x4",
      },
      {
        shaderLocation: 4,
        offset: 12 * Float32Array.BYTES_PER_ELEMENT,
        format: "float32x4",
      },
    ],
  };

  const solidRenderPipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: {
      module: shaderModule,
      entryPoint: "vs_main",
      buffers: [
        {
          arrayStride: 3 * Float32Array.BYTES_PER_ELEMENT,
          attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }],
        },
        instanceBufferLayout,
      ],
    },
    fragment: {
      module: shaderModule,
      entryPoint: "fs_solid",
      targets: [{ format: presentationFormat }],
    },
    primitive: {
      topology: "triangle-list",
      cullMode: "back",
    },
    depthStencil: {
      depthWriteEnabled: true,
      depthCompare: "less",
      format: "depth24plus",
    },
  });

  const wireframeRenderPipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: {
      module: shaderModule,
      entryPoint: "vs_main",
      buffers: [
        {
          arrayStride: 3 * Float32Array.BYTES_PER_ELEMENT,
          attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }],
        },
        instanceBufferLayout,
      ],
    },
    fragment: {
      module: shaderModule,
      entryPoint: "fs_wireframe",
      targets: [{ format: presentationFormat }],
    },
    primitive: {
      topology: "line-list",
    },
    depthStencil: {
      depthWriteEnabled: false,
      depthCompare: "less",
      format: "depth24plus",
    },
  });

  const bindGroup = device.createBindGroup({
    layout: solidRenderPipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
  });

  const projectionMatrix = mat4.create();
  const viewMatrix = mat4.create();
  const vpMatrix = mat4.create();
  const uniformBufferData = new Float32Array(
    uniformBufferSize / Float32Array.BYTES_PER_ELEMENT,
  );

  let currentCameraZ = 0.0; // This will now grow indefinitely

  const updateInstances = () => {
    // Calculate the 'normalized' Z position within the segment cycle
    // This tells us how "deep" into the current cycle the camera is.
    const cameraZInSegmentCycle = currentCameraZ % TUNNEL_SEGMENT_LENGTH;

    let instanceIdx = 0;
    const tempMat = mat4.create();
    const segmentTransform = mat4.create();

    for (let i = 0; i < NUM_RENDERING_SEGMENTS; i++) {
      // Determine the ideal, continuous Z position for this segment
      const segmentBaseZ =
        (Math.floor(currentCameraZ / TUNNEL_SEGMENT_LENGTH) -
          Math.floor(NUM_RENDERING_SEGMENTS / 2) +
          i) *
        TUNNEL_SEGMENT_LENGTH;

      // Translate the entire segment
      mat4.identity(segmentTransform);
      mat4.translate(
        segmentTransform,
        segmentTransform,
        vec3.fromValues(0, 0, segmentBaseZ),
      );
      mat4.scale(
        segmentTransform,
        segmentTransform,
        vec3.fromValues(SEGMENT_WIDTH, SEGMENT_WIDTH, SEGMENT_WIDTH),
      );

      for (let j = 0; j < numCubesPerSegment; j++) {
        mat4.multiply(tempMat, segmentTransform, baseSegmentMatrices[j]);
        currentRenderMatrices.set(tempMat, instanceIdx * 16);
        instanceIdx++;
      }
    }
    device.queue.writeBuffer(
      instanceBuffer,
      0,
      currentRenderMatrices.buffer,
      currentRenderMatrices.byteOffset,
      currentRenderMatrices.byteLength,
    );
  };

  const updateCameraAndUniforms = () => {
    const aspectRatio = canvas.width / canvas.height;
    mat4.perspective(
      projectionMatrix,
      (2 * Math.PI) / 5,
      aspectRatio,
      0.1,
      TOTAL_RENDERED_LENGTH + 50.0,
    );

    const time = Date.now() * 0.001;
    const cameraSpeed = 10.0; // Units per second

    currentCameraZ += cameraSpeed * (1 / 60); // Increment continuous Z

    const cameraPosition = vec3.fromValues(
      0.0,
      0.0,
      currentCameraZ, // Use the continuous Z directly for camera's world position
    );

    const lookAt = vec3.fromValues(0, 0, currentCameraZ - 10.0);
    const up = vec3.fromValues(0, 1, 0);
    mat4.lookAt(viewMatrix, cameraPosition, lookAt, up);

    mat4.multiply(vpMatrix, projectionMatrix, viewMatrix);

    uniformBufferData.set(vpMatrix, 0);
    uniformBufferData[16] = time;

    device.queue.writeBuffer(
      uniformBuffer,
      0,
      uniformBufferData.buffer,
      uniformBufferData.byteOffset,
      uniformBufferData.byteLength,
    );
  };

  const frame = () => {
    updateCameraAndUniforms();
    updateInstances();

    const commandEncoder = device.createCommandEncoder();
    const textureView = context.getCurrentTexture().createView();

    const renderPassDescriptor: GPURenderPassDescriptor = {
      colorAttachments: [
        {
          view: textureView,
          clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
      depthStencilAttachment: {
        view: depthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    };

    const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);

    passEncoder.setVertexBuffer(1, instanceBuffer);

    if (TOTAL_INSTANCES_IN_BUFFER > 0) {
      passEncoder.setPipeline(solidRenderPipeline);
      passEncoder.setVertexBuffer(0, triangleVertexBuffer);
      passEncoder.setIndexBuffer(triangleIndexBuffer, "uint16");
      passEncoder.setBindGroup(0, bindGroup);
      passEncoder.drawIndexed(
        cubeTriangleIndices.length,
        TOTAL_INSTANCES_IN_BUFFER,
      );

      passEncoder.setPipeline(wireframeRenderPipeline);
      passEncoder.setVertexBuffer(0, wireframeVertexBuffer);
      passEncoder.setIndexBuffer(wireframeIndexBuffer, "uint16");
      passEncoder.setBindGroup(0, bindGroup);
      passEncoder.drawIndexed(
        cubeWireframeIndices.length,
        TOTAL_INSTANCES_IN_BUFFER,
      );
    }

    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);

    requestAnimationFrame(frame);
  };

  requestAnimationFrame(frame);

  window.addEventListener("resize", () => {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    depthTexture.destroy();
    depthTexture = device.createTexture({
      size: [canvas.width, canvas.height, 1],
      format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });

    context.configure({
      device,
      format: presentationFormat,
      alphaMode: "premultiplied",
      size: [canvas.width, canvas.height],
    });
  });
}

initWebGPU();
