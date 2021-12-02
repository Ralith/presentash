use std::{collections::VecDeque, mem};

use ash::{extensions::khr, prelude::VkResult, vk, Device};

/// Manages synchronizing and rebuilding a Vulkan swapchain
pub struct Swapchain {
    options: Options,

    frames: Vec<Frame>,
    frame_index: usize,

    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
    handle: vk::SwapchainKHR,
    generation: u64,
    images: Vec<vk::Image>,
    extent: vk::Extent2D,
    format: vk::SurfaceFormatKHR,
    needs_rebuild: bool,

    old_swapchains: VecDeque<(vk::SwapchainKHR, u64)>,
}

impl Swapchain {
    /// Construct a new [`Swapchain`] for rendering at most `frames_in_flight` frames
    /// concurrently. `extent` should be the current dimensions of `surface`.
    pub fn new(
        options: Options,
        surface: vk::SurfaceKHR,
        physical_device: vk::PhysicalDevice,
        device: &Device,
        frames_in_flight: usize,
        extent: vk::Extent2D,
    ) -> Self {
        Self {
            options,

            frames: (0..frames_in_flight)
                .map(|_| unsafe {
                    Frame {
                        complete: device
                            .create_fence(
                                &vk::FenceCreateInfo::builder()
                                    .flags(vk::FenceCreateFlags::SIGNALED),
                                None,
                            )
                            .unwrap(),
                        acquire: device
                            .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                            .unwrap(),
                        generation: 0,
                    }
                })
                .collect(),
            frame_index: 0,

            surface,
            physical_device,
            handle: vk::SwapchainKHR::null(),
            generation: 0,
            images: Vec::new(),
            extent,
            format: vk::SurfaceFormatKHR::default(),
            needs_rebuild: true,

            old_swapchains: VecDeque::new(),
        }
    }

    /// Destroy all swapchain resources. Must not be called while any frames are still in flight on
    /// the GPU.
    pub unsafe fn destroy(&mut self, device: &Device, swapchain_fn: &khr::Swapchain) {
        for frame in &self.frames {
            device.destroy_fence(frame.complete, None);
            device.destroy_semaphore(frame.acquire, None);
        }
        if self.handle != vk::SwapchainKHR::null() {
            swapchain_fn.destroy_swapchain(self.handle, None);
        }
        for &(swapchain, _) in &self.old_swapchains {
            swapchain_fn.destroy_swapchain(swapchain, None);
        }
    }

    /// Force the swapchain to be rebuilt on the next [`acquire`](Self::acquire) call, passing in
    /// the surface's current size.
    pub fn update(&mut self, extent: vk::Extent2D) {
        self.extent = extent;
        self.needs_rebuild = true;
    }

    /// Maximum number of frames that may be concurrently rendered
    pub fn frames_in_flight(&self) -> usize {
        self.frames.len()
    }

    /// Latest set of swapchain images, keyed by [`AcquiredImage::image_index`]
    pub fn images(&self) -> &[vk::Image] {
        &self.images
    }

    /// Format of images in [`images`](Self::images), and the color space that will be used to
    /// present them
    pub fn format(&self) -> vk::SurfaceFormatKHR {
        self.format
    }

    /// Dimensions of images in [`images`](Self::images)
    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }

    /// Acquire a swapchain image
    ///
    /// Returns [`vk::Result::ERROR_OUT_OF_DATE_KHR`] iff the configured format or present modes
    /// could not be satisfied.
    pub unsafe fn acquire(
        &mut self,
        device: &Device,
        surface_fn: &khr::Surface,
        swapchain_fn: &khr::Swapchain,
    ) -> VkResult<AcquiredImage> {
        let frame_index = self.frame_index;
        let next_frame_index = (self.frame_index + 1) % self.frames.len();
        let frame = &self.frames[frame_index];
        let acquire = frame.acquire;
        device.wait_for_fences(&[frame.complete], true, !0)?;

        // Destroy swapchains that are guaranteed not to be in use now that this frame has finished
        while let Some(&(swapchain, generation)) = self.old_swapchains.front() {
            if self.frames[next_frame_index].generation == generation {
                break;
            }
            swapchain_fn.destroy_swapchain(swapchain, None);
            self.old_swapchains.pop_front();
        }

        loop {
            if !self.needs_rebuild {
                match swapchain_fn.acquire_next_image(self.handle, !0, acquire, vk::Fence::null()) {
                    Ok((index, suboptimal)) => {
                        self.needs_rebuild = suboptimal;
                        self.frames[frame_index].generation = self.generation;
                        self.frame_index = next_frame_index;
                        device
                            .reset_fences(&[self.frames[frame_index].complete])
                            .unwrap();
                        return Ok(AcquiredImage {
                            image_index: index as usize,
                            frame_index,
                            ready: acquire,
                            complete: self.frames[frame_index].complete,
                            generation: self.generation,
                        });
                    }
                    Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {}
                    Err(e) => return Err(e),
                }
            };

            // Rebuild swapchain
            let surface_capabilities = surface_fn
                .get_physical_device_surface_capabilities(self.physical_device, self.surface)
                .unwrap();
            self.extent = match surface_capabilities.current_extent.width {
                // If Vulkan doesn't know, the windowing system probably does. Known to apply at
                // least to Wayland.
                std::u32::MAX => vk::Extent2D {
                    width: self.extent.width,
                    height: self.extent.height,
                },
                _ => surface_capabilities.current_extent,
            };
            let pre_transform = if surface_capabilities
                .supported_transforms
                .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
            {
                vk::SurfaceTransformFlagsKHR::IDENTITY
            } else {
                surface_capabilities.current_transform
            };
            let present_mode = surface_fn
                .get_physical_device_surface_present_modes(self.physical_device, self.surface)?
                .iter()
                .filter_map(|&mode| {
                    Some((
                        mode,
                        self.options
                            .present_mode_preference
                            .iter()
                            .position(|&pref| pref == mode)?,
                    ))
                })
                .min_by_key(|&(_, priority)| priority)
                .map(|(mode, _)| mode)
                .ok_or(vk::Result::ERROR_OUT_OF_DATE_KHR)?;

            let desired_image_count =
                (surface_capabilities.min_image_count + 1).max(self.frames.len() as u32);
            let image_count = if surface_capabilities.max_image_count > 0 {
                surface_capabilities
                    .max_image_count
                    .min(desired_image_count)
            } else {
                desired_image_count
            };

            self.format = surface_fn
                .get_physical_device_surface_formats(self.physical_device, self.surface)?
                .iter()
                .filter_map(|&format| {
                    Some((
                        format,
                        self.options
                            .format_preference
                            .iter()
                            .position(|&pref| pref == format),
                    ))
                })
                .min_by_key(|&(_, priority)| priority)
                .map(|(mode, _)| mode)
                .ok_or(vk::Result::ERROR_OUT_OF_DATE_KHR)?;

            if self.handle != vk::SwapchainKHR::null() {
                self.old_swapchains
                    .push_back((self.handle, self.generation));
            }
            let handle = swapchain_fn.create_swapchain(
                &vk::SwapchainCreateInfoKHR::builder()
                    .surface(self.surface)
                    .min_image_count(image_count)
                    .image_color_space(self.format.color_space)
                    .image_format(self.format.format)
                    .image_extent(self.extent)
                    .image_usage(self.options.usage)
                    .image_sharing_mode(self.options.sharing_mode)
                    .pre_transform(pre_transform)
                    .composite_alpha(self.options.composite_alpha)
                    .present_mode(present_mode)
                    .clipped(true)
                    .image_array_layers(1)
                    .old_swapchain(mem::replace(&mut self.handle, vk::SwapchainKHR::null())),
                None,
            )?;
            self.generation = self.generation.wrapping_add(1);
            self.handle = handle;
            self.images = swapchain_fn.get_swapchain_images(handle)?;
            self.needs_rebuild = false;
        }
    }

    /// Queue presentation of a previously acquired image
    pub unsafe fn queue_present(
        &mut self,
        swapchain_fn: &khr::Swapchain,
        queue: vk::Queue,
        render_complete: vk::Semaphore,
        image_index: usize,
    ) -> VkResult<()> {
        match swapchain_fn.queue_present(
            queue,
            &vk::PresentInfoKHR::builder()
                .wait_semaphores(&[render_complete])
                .swapchains(&[self.handle])
                .image_indices(&[image_index as u32]),
        ) {
            Ok(false) => Ok(()),
            Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.needs_rebuild = true;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }
}

/// [`Swapchain`] configuration
#[derive(Debug, Clone)]
pub struct Options {
    format_preference: Vec<vk::SurfaceFormatKHR>,
    present_mode_preference: Vec<vk::PresentModeKHR>,
    usage: vk::ImageUsageFlags,
    sharing_mode: vk::SharingMode,
    composite_alpha: vk::CompositeAlphaFlagsKHR,
}

impl Options {
    pub fn new() -> Self {
        Self::default()
    }

    /// Preference-ordered list of image formats and color spaces. Defaults to 8-bit sRGB.
    pub fn format_preference(&mut self, formats: &[vk::SurfaceFormatKHR]) -> &mut Self {
        self.format_preference = formats.into();
        self
    }

    /// Preference-ordered list of presentation modes. Defaults to [`vk::PresentModeKHR::FIFO`].
    pub fn present_mode_preference(&mut self, modes: &[vk::PresentModeKHR]) -> &mut Self {
        self.present_mode_preference = modes.into();
        self
    }

    /// Required swapchain image usage flags. Defaults to [`vk::ImageUsageFlags::COLOR_ATTACHMENT`].
    pub fn usage(&mut self, usage: vk::ImageUsageFlags) -> &mut Self {
        self.usage = usage;
        self
    }

    /// Requires swapchain image sharing mode. Defaults to [`vk::SharingMode::EXCLUSIVE`].
    pub fn sharing_mode(&mut self, mode: vk::SharingMode) -> &mut Self {
        self.sharing_mode = mode;
        self
    }

    /// Requires swapchain image composite alpha. Defaults to [`vk::CompositeAlphaFlagsKHR::OPAQUE`].
    pub fn composite_alpha(&mut self, value: vk::CompositeAlphaFlagsKHR) -> &mut Self {
        self.composite_alpha = value;
        self
    }
}

impl Default for Options {
    fn default() -> Self {
        Self {
            format_preference: vec![
                vk::SurfaceFormatKHR {
                    format: vk::Format::B8G8R8A8_SRGB,
                    color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
                },
                vk::SurfaceFormatKHR {
                    format: vk::Format::R8G8B8A8_SRGB,
                    color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
                },
            ],
            present_mode_preference: vec![vk::PresentModeKHR::FIFO],
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
        }
    }
}

struct Frame {
    complete: vk::Fence,
    acquire: vk::Semaphore,
    generation: u64,
}

/// Result of a successful [`Swapchain::acquire`] call
#[derive(Debug, Copy, Clone)]
pub struct AcquiredImage {
    /// Index of the image to write to in [`Swapchain::images`]
    pub image_index: usize,
    /// Index of the frame in flight, for use tracking your own per-frame resources, which may be
    /// accessed immediately after [`Swapchain::acquire`] returns
    pub frame_index: usize,
    /// Must be waited on before accessing the image associated with `image_index`
    pub ready: vk::Semaphore,
    /// Must be signaled when access to the image associated with `image_index` and any per-frame
    /// resources associated with `frame_index` is complete
    pub complete: vk::Fence,
    /// Incremented whenever [`Swapchain::images`] has, and [`Swapchain::extent`] and
    /// [`Swapchain::format`] may have, changed. Initially 0.
    pub generation: u64,
}
