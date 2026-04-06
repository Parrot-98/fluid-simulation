use std::f32::consts::PI;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

const CIRCLE_RADIUS: f32 = 0.01;
const BALL_COUNT: usize = 100000;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 2],
    pub color: [f32; 3],
}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x3];

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceData {
    pub offset: [f32; 2],
}

impl InstanceData {
    const ATTRIBS: [wgpu::VertexAttribute; 1] =
        wgpu::vertex_attr_array![2 => Float32x2];

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<InstanceData>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &Self::ATTRIBS,
        }
    }
}

pub struct Ball {
    pub position: [f32; 2],
    pub velocity: [f32; 2],
}

fn build_circle(
    segments: u32,
    radius: f32,
    aspect: f32,
    cx: f32,
    cy: f32,
    color: [f32; 3],
) -> (Vec<Vertex>, Vec<u16>) {
    let mut vertices: Vec<Vertex> = Vec::with_capacity((segments + 2) as usize);
    let mut indices: Vec<u16> = Vec::with_capacity((segments * 3) as usize);

    vertices.push(Vertex { position: [cx, cy], color });

    for i in 0..=segments {
        let angle = (i as f32 / segments as f32) * std::f32::consts::TAU;
        vertices.push(Vertex {
            position: [cx + (angle.cos() * radius) / aspect, cy + angle.sin() * radius],
            color,
        });
    }

    for i in 0..segments {
        indices.push(0);
        indices.push((i + 1) as u16);
        indices.push((i + 2) as u16);
    }

    (vertices, indices)
}

const SHADER_SRC: &str = r#"
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) color:    vec3<f32>,
}

struct InstanceInput {
    @location(2) offset: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0)       color:         vec3<f32>,
}

@vertex
fn vs_main(in: VertexInput, inst: InstanceInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(in.position + inst.offset, 0.0, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
"#;

pub struct State {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pub size: PhysicalSize<u32>,

    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    instance_buffer: wgpu::Buffer,
    num_indices: u32,
}

impl State {
    pub async fn new(window: Window) -> Self {
        let window = Arc::new(window);
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::DX12,
            flags: wgpu::InstanceFlags::empty(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            backend_options: wgpu::BackendOptions::default(),
            display: None,
        });

        let surface = instance
            .create_surface(Arc::clone(&window))
            .expect("Failed to create surface");

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find adapter");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            })
            .await
            .expect("Failed to create device");

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Circle Shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[],
            immediate_size: 0,
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Circle Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc(), InstanceData::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // Temporary placeholder vertex/index buffers — replaced by rebuild_buffers()
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&[Vertex { position: [0.0, 0.0], color: [0.0; 3] }]),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&[0u16]),
            usage: wgpu::BufferUsages::INDEX,
        });

        // Instance buffer — holds one offset per ball, updated every frame
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instance Buffer"),
            size: (BALL_COUNT * std::mem::size_of::<InstanceData>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut state = Self {
            window,
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            instance_buffer,
            num_indices: 0,
        };

        state.rebuild_buffers();
        state
    }

    pub fn rebuild_buffers(&mut self) {
        let aspect = self.size.width as f32 / self.size.height as f32;
        let (all_vertices, all_indices) =
            build_circle(64, CIRCLE_RADIUS, aspect, 0.0, 0.0, [0.2, 0.4, 1.0]);

        self.vertex_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&all_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        self.index_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&all_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        self.num_indices = all_indices.len() as u32;
    }

    pub fn update_instances(&mut self, balls: &[Ball]) {
        let instances: Vec<InstanceData> = balls
            .iter()
            .map(|b| InstanceData { offset: b.position })
            .collect();
        self.queue.write_buffer(
            &self.instance_buffer,
            0,
            bytemuck::cast_slice(&instances),
        );
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.rebuild_buffers();
        }
    }

    pub fn reconfigure(&mut self) {
        self.resize(self.size);
    }

    pub fn render(&mut self) -> bool {
        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) => frame,
            wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
                self.reconfigure();
                frame
            }
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded => return false,
            wgpu::CurrentSurfaceTexture::Outdated
            | wgpu::CurrentSurfaceTexture::Lost => {
                self.reconfigure();
                return false;
            }
            wgpu::CurrentSurfaceTexture::Validation => return false,
        };

        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.05,
                            g: 0.05,
                            b: 0.05,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..self.num_indices, 0, 0..BALL_COUNT as u32);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        true
    }

    pub fn window(&self) -> &Window {
        &self.window
    }
}

pub struct App {
    state: Option<State>,
    gravity_on: bool,
    balls: Vec<Ball>,
    last_frame: std::time::Instant,
}

const SMOOTHING_RADIUS: f32 = 0.2;
const MASS: f32 = 1.0;

fn smoothing_kernel(radius: f32, dst: f32) -> f32 {
    let value = f32::max(0.0, radius * radius - dst * dst);
    value * value * value
}

fn calculate_density(sample_point: [f32; 2], positions: &[[f32; 2]]) -> f32 {
    let mut density = 0.0;

    for position in positions {
        let dx = position[0] - sample_point[0];
        let dy = position[1] - sample_point[1];
        let dst = (dx * dx + dy * dy).sqrt();  // magnitude
        let influence = smoothing_kernel(SMOOTHING_RADIUS, dst);
        density += MASS * influence;
    }

    density
}


const SPACING: f32 = 0.02;
const NUMBER_BALLS_SIDE: u32 = 50;

impl Default for App {
    fn default() -> Self {
        let mut balls = Vec::new();
        for row in 0..NUMBER_BALLS_SIDE {
            for col in 0..NUMBER_BALLS_SIDE {
                balls.push(Ball {
                    position: [
                        -0.9 + col as f32 * SPACING,
                        0.9 - row as f32 * SPACING,
                    ],
                    velocity: [0.0, 0.0],
                });
            }
        }
        Self {
            state: None,
            gravity_on: false,
            balls,
            last_frame: std::time::Instant::now(),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            let window = event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("wgpu circles")
                        .with_inner_size(winit::dpi::LogicalSize::new(1000.0, 800.0))
                )
                .unwrap();
            self.state = Some(pollster::block_on(State::new(window)));
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = self.state.as_mut() else { return };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(new_size) => state.resize(new_size),

            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    physical_key: PhysicalKey::Code(key),
                    state: ElementState::Pressed,
                    ..
                },
                ..
            } => match key {
                KeyCode::Escape => event_loop.exit(),
                KeyCode::KeyG => self.gravity_on = !self.gravity_on,
                KeyCode::Space => {
                    if self.gravity_on {
                        for ball in &mut self.balls {
                            ball.velocity[1] = 1.5;
                        }
                    }
                }
                _ => {}
            },

            WindowEvent::RedrawRequested => {
                let now = std::time::Instant::now();
                let dt = now.duration_since(self.last_frame).as_secs_f32();
                self.last_frame = now;

                if self.gravity_on {
                    const GRAVITY: f32 = 4.0;
                    let floor_limit = -1.0 + CIRCLE_RADIUS;

                    for ball in &mut self.balls {
                        ball.velocity[1] -= GRAVITY * dt;
                        let new_y = ball.position[1] + ball.velocity[1] * dt;

                        if new_y < floor_limit {
                            ball.position[1] = floor_limit;
                            ball.velocity[1] *= -0.5;
                        } else {
                            ball.position[1] = new_y;
                        }
                    }
                }

                state.update_instances(&self.balls);
                state.render();
                state.window().request_redraw();
            }

            _ => {}
        }
    }
}

pub fn run() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();
}