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

// --- Constants ---
const CIRCLE_RADIUS: f32 = 0.015;
const BALL_COUNT: usize = 1600;
const SMOOTHING_RADIUS: f32 = 0.2;
const MASS: f32 = 1.0;
const TARGET_DENSITY: f32 = 0.0000001;
const PRESSURE_MULTIPLIER: f32 = 500.0;
const GRAVITY: f32 = 2.81;
const VISCOSITY_STRENGTH: f32 = 10.0;

// --- Types ---
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 2],
    pub color: [f32; 3],
}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x3];
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
    const ATTRIBS: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![2 => Float32x2];
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


fn build_circle(segments: u32, radius: f32, aspect: f32, color: [f32; 3]) -> (Vec<Vertex>, Vec<u16>) {
    let mut vertices = Vec::with_capacity((segments + 2) as usize);
    let mut indices = Vec::with_capacity((segments * 3) as usize);

    vertices.push(Vertex { position: [0.0, 0.0], color });

    for i in 0..=segments {
        let angle = (i as f32 / segments as f32) * std::f32::consts::TAU;
        // Apply aspect ratio correction to the vertex positions
        vertices.push(Vertex {
            position: [(angle.cos() * radius) / aspect, angle.sin() * radius],
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

// Physics 
fn smoothing_kernel(radius: f32, dst: f32) -> f32 {
    if dst >= radius { return 0.0; }
    let value = radius * radius - dst * dst;
    value * value * value
}

fn smoothing_kernel_derivative(radius: f32, dst: f32) -> f32 {
    if dst >= radius || dst < 0.00001 { return 0.0; }
    let scale = -6.0 * dst;
    let v = radius * radius - dst * dst;
    scale * v * v
}

fn calculate_density(sample_point: [f32; 2], positions: &[[f32; 2]]) -> f32 {
    let mut density = 0.0;
    for position in positions {
        let dx = position[0] - sample_point[0];
        let dy = position[1] - sample_point[1];
        let dst = (dx * dx + dy * dy).sqrt();
        density += MASS * smoothing_kernel(SMOOTHING_RADIUS, dst);
    }
    density
}

fn density_to_pressure(density: f32) -> f32 {
    f32::max(0.0, (density - TARGET_DENSITY) * PRESSURE_MULTIPLIER)
}

fn calculate_pressure_force(ball_idx: usize, positions: &[[f32; 2]], densities: &[f32]) -> [f32; 2] {
    let mut force = [0.0f32; 2];
    let p_idx = positions[ball_idx];
    let rho_idx = densities[ball_idx];
    let pres_idx = density_to_pressure(rho_idx);

    for i in 0..positions.len() {
        if i == ball_idx { continue; }
        let dx = positions[i][0] - p_idx[0];
        let dy = positions[i][1] - p_idx[1];
        let dst = (dx * dx + dy * dy).sqrt();

        if dst < SMOOTHING_RADIUS && dst > 0.0001 {
            let nx = dx / dst;
            let ny = dy / dst;
            let slope = smoothing_kernel_derivative(SMOOTHING_RADIUS, dst);
            let shared_pressure = (pres_idx + density_to_pressure(densities[i])) / 2.0;
            
            // Standard SPH Pressure Gradient
            force[0] += shared_pressure * nx * slope * MASS / densities[i];
            force[1] += shared_pressure * ny * slope * MASS / densities[i];
        }
    }
    force
}

fn calculate_viscosity_force(ball_idx: usize, balls: &[Ball], densities: &[f32]) -> [f32; 2] {
    let mut viscosity_force = [0.0f32; 2];
    let p_idx = balls[ball_idx].position;
    let v_idx = balls[ball_idx].velocity;

    for i in 0..balls.len() {
        if i == ball_idx { continue; }
        
        let dx = balls[i].position[0] - p_idx[0];
        let dy = balls[i].position[1] - p_idx[1];
        let dst = (dx * dx + dy * dy).sqrt();

        if dst < SMOOTHING_RADIUS && dst > 0.0001 {
            let weight = smoothing_kernel(SMOOTHING_RADIUS, dst);
            
            // Relative velocity
            let rel_v_x = balls[i].velocity[0] - v_idx[0];
            let rel_v_y = balls[i].velocity[1] - v_idx[1];

            viscosity_force[0] += VISCOSITY_STRENGTH * MASS * rel_v_x * weight / densities[i];
            viscosity_force[1] += VISCOSITY_STRENGTH * MASS * rel_v_y * weight / densities[i];
        }
    }
    viscosity_force
}

// --- Shader ---
const SHADER_SRC: &str = r#"
struct VertexInput { @location(0) position: vec2<f32>, @location(1) color: vec3<f32> }
struct InstanceInput { @location(2) offset: vec2<f32> }
struct VertexOutput { @builtin(position) clip_pos: vec4<f32>, @location(0) color: vec3<f32> }

@vertex
fn vs_main(in: VertexInput, inst: InstanceInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_pos = vec4<f32>(in.position + inst.offset, 0.0, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
"#;

// --- WGPU State ---
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
        let instance = wgpu::Instance::default(); 

        let surface = instance.create_surface(Arc::clone(&window)).unwrap();
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            ..Default::default()
        }).await.unwrap();

        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default()).await.unwrap();
        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats.iter().copied().find(|f| f.is_srgb()).unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: None,
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
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 1024, usage: wgpu::BufferUsages::VERTEX, mapped_at_creation: false });
        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 1024, usage: wgpu::BufferUsages::INDEX, mapped_at_creation: false });
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instance Buffer"),
            size: (BALL_COUNT * std::mem::size_of::<InstanceData>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut state = Self { window, surface, device, queue, config, size, render_pipeline, vertex_buffer, index_buffer, instance_buffer, num_indices: 0 };
        state.rebuild_buffers();
        state
    }

    pub fn rebuild_buffers(&mut self) {
        let aspect = self.size.width as f32 / self.size.height as f32;
        let (v, i) = build_circle(32, CIRCLE_RADIUS, aspect, [0.3, 0.6, 1.0]);
        self.vertex_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(&v), usage: wgpu::BufferUsages::VERTEX,
        });
        self.index_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(&i), usage: wgpu::BufferUsages::INDEX,
        });
        self.num_indices = i.len() as u32;
    }

    pub fn render(&mut self, balls: &[Ball]) {
        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) => frame,
            wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            _ => return,
        };
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                ..Default::default()
            });
            rp.set_pipeline(&self.render_pipeline);
            rp.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            rp.set_vertex_buffer(1, self.instance_buffer.slice(..));
            rp.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            // Draw only the current number of balls
            rp.draw_indexed(0..self.num_indices, 0, 0..balls.len() as u32);
        }

        let instances: Vec<InstanceData> = balls.iter().map(|b| InstanceData { offset: b.position }).collect();
        self.queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&instances));
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }
}

// --- Application Logic ---
pub struct App {
    state: Option<State>,
    gravity_on: bool,
    balls: Vec<Ball>,
    last_frame: std::time::Instant,
}

impl Default for App {
    fn default() -> Self {

        //randomize
        use rand::Rng; 
        let mut rng = rand::thread_rng();
        let mut balls = Vec::new();

        let ball_count = 900;
        
        let spawn_range = -0.8..0.8; 

        for _ in 0..ball_count {
            balls.push(Ball {
                position: [
                    rng.gen_range(spawn_range.clone()), 
                    rng.gen_range(spawn_range.clone()), 
                ],
                velocity: [0.0, 0.0],
            });
        }

        Self { 
            state: None, 
            gravity_on: false, 
            balls, 
            last_frame: std::time::Instant::now() 
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
    if self.state.is_none() {
        use winit::dpi::LogicalSize;
        let window = event_loop.create_window(Window::default_attributes().with_title("SPH Fluid").with_inner_size(LogicalSize::new(1200.0, 800.0))).unwrap();
        let state = pollster::block_on(State::new(window));
        state.window.request_redraw();  
        self.state = Some(state);
    }
}

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let state = self.state.as_mut().unwrap();
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(s) => { state.config.width = s.width; state.config.height = s.height; state.surface.configure(&state.device, &state.config); state.size = s; state.rebuild_buffers(); }
            WindowEvent::KeyboardInput { event: KeyEvent { physical_key: PhysicalKey::Code(key), state: ElementState::Pressed, .. }, .. } => match key {
                KeyCode::KeyG => self.gravity_on = !self.gravity_on,
                KeyCode::Space => self.balls.iter_mut().for_each(|b| b.velocity[1] = 2.0),
                _ => {}
            },
            WindowEvent::RedrawRequested => {
                let now = std::time::Instant::now();
                let dt = now.duration_since(self.last_frame).as_secs_f32().min(0.016);
                self.last_frame = now;

                let positions: Vec<[f32; 2]> = self.balls.iter().map(|b| b.position).collect();
                let densities: Vec<f32> = positions.iter().map(|&p| calculate_density(p, &positions)).collect();

                for i in 0..self.balls.len() {
                    let pressure = calculate_pressure_force(i, &positions, &densities);
                    let viscosity = calculate_viscosity_force(i, &self.balls, &densities);
                    self.balls[i].velocity[0] += pressure[0] + viscosity[0] *  dt;
                    self.balls[i].velocity[1] += pressure[1] + viscosity[1] * dt;
                    if self.gravity_on { self.balls[i].velocity[1] -= GRAVITY * dt; }
                }

                for ball in &mut self.balls {
                    ball.position[0] += ball.velocity[0] * dt;
                    ball.position[1] += ball.velocity[1] * dt;

                    let bounce = -0.5;
                    let margin = 1.0 - CIRCLE_RADIUS;
                    if ball.position[1].abs() > margin {
                        ball.position[1] = margin * ball.position[1].signum();
                        ball.velocity[1] *= bounce;
                    }
                    if ball.position[0].abs() > margin {
                        ball.position[0] = margin * ball.position[0].signum();
                        ball.velocity[0] *= bounce;
                    }
                }

                state.render(&self.balls);
                state.window.request_redraw();
            }
            _ => {}
        }
    }
}

pub fn run() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();
}