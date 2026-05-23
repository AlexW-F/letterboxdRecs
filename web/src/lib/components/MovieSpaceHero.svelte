<script lang="ts">
	import { onDestroy, onMount } from 'svelte';
	import { fetchBackgroundCoords, type BackgroundScatter } from '$lib/api';

	let {
		height = '70vh',
		points = 4500
	}: {
		height?: string;
		points?: number;
	} = $props();

	let container: HTMLDivElement;
	let loading = $state(true);
	let error = $state<string | null>(null);
	let hoveredTitle = $state<string | null>(null);
	let hoveredGenre = $state<string | null>(null);

	let cleanup: (() => void) | null = null;

	onMount(async () => {
		let data: BackgroundScatter;
		try {
			data = await fetchBackgroundCoords(points);
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
			loading = false;
			return;
		}

		const THREE = await import('three');

		// ---------------------------------------------------------------
		// Scene + renderer
		// ---------------------------------------------------------------
		const renderer = new THREE.WebGLRenderer({
			alpha: true,
			antialias: true,
			powerPreference: 'high-performance'
		});
		renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
		renderer.setSize(container.clientWidth, container.clientHeight, false);
		// Background transparent so the page gradients show through.
		renderer.setClearColor(0x000000, 0);
		container.appendChild(renderer.domElement);
		renderer.domElement.style.cssText =
			'width: 100%; height: 100%; display: block; touch-action: none; cursor: grab;';

		const scene = new THREE.Scene();
		// Fog the same color as the page background so distant points dissolve
		// into the void instead of popping at the far plane.
		scene.fog = new THREE.Fog(0x0a0c10, 18, 80);

		const camera = new THREE.PerspectiveCamera(
			72,
			container.clientWidth / container.clientHeight,
			0.1,
			200
		);
		camera.position.set(0, 0, 0);

		// ---------------------------------------------------------------
		// Center + scale the cloud so the camera path stays inside it.
		// ---------------------------------------------------------------
		const raw = data.coords;
		let cx = 0, cy = 0, cz = 0;
		for (const [x, y, z] of raw) {
			cx += x;
			cy += y;
			cz += z;
		}
		cx /= raw.length;
		cy /= raw.length;
		cz /= raw.length;

		let maxR = 0;
		for (const [x, y, z] of raw) {
			const dx = x - cx,
				dy = y - cy,
				dz = z - cz;
			const r2 = dx * dx + dy * dy + dz * dz;
			if (r2 > maxR) maxR = r2;
		}
		maxR = Math.sqrt(maxR);
		const scale = 28 / maxR; // cloud radius ~28 units around origin

		const positions = new Float32Array(raw.length * 3);
		const colors = new Float32Array(raw.length * 3);
		const sizes = new Float32Array(raw.length);

		const c = new THREE.Color();
		// Slight per-point size variation by popularity gives the cloud a
		// near-field / far-field star feel; popular films are visibly larger.
		const popMax = Math.max(...data.popularity);
		for (let i = 0; i < raw.length; i++) {
			positions[3 * i] = (raw[i][0] - cx) * scale;
			positions[3 * i + 1] = (raw[i][1] - cy) * scale;
			positions[3 * i + 2] = (raw[i][2] - cz) * scale;
			const hex = data.genre_colors[data.genres[i]] ?? '#cccccc';
			c.set(hex);
			colors[3 * i] = c.r;
			colors[3 * i + 1] = c.g;
			colors[3 * i + 2] = c.b;
			// Size scales loosely with log(popularity); range ~0.7–2.2.
			const popN = Math.log1p(data.popularity[i]) / Math.log1p(popMax);
			sizes[i] = 0.7 + popN * 1.5;
		}

		const geometry = new THREE.BufferGeometry();
		geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
		geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
		geometry.setAttribute('aSize', new THREE.BufferAttribute(sizes, 1));

		// ---------------------------------------------------------------
		// Soft circular sprite via shader. Additive blending so overlaps
		// glow brighter — like the milky way.
		// ---------------------------------------------------------------
		const starMaterial = new THREE.ShaderMaterial({
			uniforms: {
				uPixelRatio: { value: Math.min(window.devicePixelRatio, 2) },
				uSize: { value: 38.0 },
				uTime: { value: 0 }
			},
			vertexShader: `
				attribute float aSize;
				varying vec3 vColor;
				varying float vDist;
				uniform float uPixelRatio;
				uniform float uSize;
				void main() {
					vColor = color;
					vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
					vDist = -mvPosition.z;
					gl_Position = projectionMatrix * mvPosition;
					gl_PointSize = uSize * aSize * uPixelRatio / -mvPosition.z;
				}
			`,
			fragmentShader: `
				varying vec3 vColor;
				varying float vDist;
				void main() {
					vec2 d = gl_PointCoord - vec2(0.5);
					float r2 = dot(d, d);
					if (r2 > 0.25) discard;
					float alpha = smoothstep(0.25, 0.0, r2);
					// Slight depth-based brightness — closer stars feel more alive.
					float depthFade = 1.0 - smoothstep(20.0, 70.0, vDist);
					gl_FragColor = vec4(vColor, alpha * (0.65 + 0.35 * depthFade));
				}
			`,
			blending: THREE.AdditiveBlending,
			depthWrite: false,
			transparent: true,
			vertexColors: true
		});

		const stars = new THREE.Points(geometry, starMaterial);
		scene.add(stars);

		// A second, smaller "core glow" pass — same points rendered tinier with
		// stronger additive blending creates the soft halo around bright stars.
		const haloMaterial = new THREE.PointsMaterial({
			size: 0.85,
			vertexColors: true,
			transparent: true,
			opacity: 0.55,
			blending: THREE.AdditiveBlending,
			depthWrite: false,
			sizeAttenuation: true
		});
		const halo = new THREE.Points(geometry, haloMaterial);
		scene.add(halo);

		// ---------------------------------------------------------------
		// Pick: raycaster for hover tooltips
		// ---------------------------------------------------------------
		const raycaster = new THREE.Raycaster();
		// Threshold (in world units) makes screen-space hover forgiving.
		raycaster.params.Points = { threshold: 0.65 };
		const pointer = new THREE.Vector2(-2, -2); // off-screen by default

		// ---------------------------------------------------------------
		// Interaction state
		// ---------------------------------------------------------------
		let isDragging = false;
		let dragX = 0;
		let dragY = 0;
		// User-applied look rotations on top of the auto camera path.
		let yaw = 0;
		let pitch = 0;
		// Pointer-following parallax (subtle).
		let parallaxX = 0;
		let parallaxY = 0;
		let targetParallaxX = 0;
		let targetParallaxY = 0;

		function onPointerDown(e: PointerEvent) {
			isDragging = true;
			dragX = e.clientX;
			dragY = e.clientY;
			renderer.domElement.style.cursor = 'grabbing';
			(e.target as HTMLElement).setPointerCapture?.(e.pointerId);
		}
		function onPointerUp(e: PointerEvent) {
			isDragging = false;
			renderer.domElement.style.cursor = 'grab';
			(e.target as HTMLElement).releasePointerCapture?.(e.pointerId);
		}
		function onPointerMove(e: PointerEvent) {
			const rect = renderer.domElement.getBoundingClientRect();
			const nx = (e.clientX - rect.left) / rect.width;
			const ny = (e.clientY - rect.top) / rect.height;
			pointer.x = nx * 2 - 1;
			pointer.y = -(ny * 2 - 1);

			targetParallaxX = pointer.x * 0.25;
			targetParallaxY = pointer.y * 0.18;

			if (isDragging) {
				const dx = e.clientX - dragX;
				const dy = e.clientY - dragY;
				yaw -= dx * 0.005;
				pitch -= dy * 0.004;
				// Clamp pitch so we don't flip.
				pitch = Math.max(-0.85, Math.min(0.85, pitch));
				dragX = e.clientX;
				dragY = e.clientY;
			}
		}
		function onPointerLeave() {
			pointer.set(-2, -2);
			targetParallaxX = 0;
			targetParallaxY = 0;
			hoveredTitle = null;
			hoveredGenre = null;
		}

		renderer.domElement.addEventListener('pointerdown', onPointerDown);
		window.addEventListener('pointerup', onPointerUp);
		renderer.domElement.addEventListener('pointermove', onPointerMove);
		renderer.domElement.addEventListener('pointerleave', onPointerLeave);

		// ---------------------------------------------------------------
		// Resize
		// ---------------------------------------------------------------
		const resize = () => {
			const w = container.clientWidth;
			const h = container.clientHeight;
			camera.aspect = w / h;
			camera.updateProjectionMatrix();
			renderer.setSize(w, h, false);
			starMaterial.uniforms.uPixelRatio.value = Math.min(window.devicePixelRatio, 2);
		};
		const ro = new ResizeObserver(resize);
		ro.observe(container);

		// ---------------------------------------------------------------
		// Animation loop
		// ---------------------------------------------------------------
		const start = performance.now();
		let raf = 0;
		let lastPickTime = 0;

		const tmpVec = new THREE.Vector3();
		const lookTarget = new THREE.Vector3();

		const animate = () => {
			const t = (performance.now() - start) / 1000;
			starMaterial.uniforms.uTime.value = t;

			// Cinematic Lissajous-ish drift through the cloud — the camera lives
			// near the cloud's center but slowly weaves around so the parallax
			// reveals new stars as you watch.
			const driftR = 8;
			camera.position.x = driftR * Math.cos(t * 0.06);
			camera.position.y = driftR * 0.35 * Math.sin(t * 0.09);
			camera.position.z = driftR * Math.sin(t * 0.07);

			// Smoothed parallax + drag-look toward (yaw, pitch).
			parallaxX += (targetParallaxX - parallaxX) * 0.05;
			parallaxY += (targetParallaxY - parallaxY) * 0.05;

			// Look slightly ahead of the path so we feel motion, biased by
			// user drag and pointer parallax.
			lookTarget.set(
				camera.position.x * 0.3 + (yaw + parallaxX) * 14,
				camera.position.y * 0.3 + (pitch + parallaxY) * 10,
				camera.position.z * 0.3 + 4 * Math.cos(t * 0.05)
			);
			camera.lookAt(lookTarget);

			// Spin the cloud itself, very slowly — adds passive motion when the
			// user is hovering and the camera path is paused-ish.
			stars.rotation.y = t * 0.012;
			halo.rotation.y = t * 0.012;

			// Picking: throttle to ~30 Hz so this doesn't blow CPU on big clouds.
			if (performance.now() - lastPickTime > 33 && pointer.x > -1.5) {
				lastPickTime = performance.now();
				raycaster.setFromCamera(pointer, camera);
				const hits = raycaster.intersectObject(stars, false);
				if (hits.length > 0) {
					const idx = hits[0].index ?? -1;
					if (idx >= 0 && idx < data.titles.length) {
						hoveredTitle = data.titles[idx];
						hoveredGenre = data.genres[idx];
					}
				} else {
					hoveredTitle = null;
					hoveredGenre = null;
				}
			}

			renderer.render(scene, camera);
			raf = requestAnimationFrame(animate);
		};
		animate();
		loading = false;

		cleanup = () => {
			cancelAnimationFrame(raf);
			ro.disconnect();
			renderer.domElement.removeEventListener('pointerdown', onPointerDown);
			window.removeEventListener('pointerup', onPointerUp);
			renderer.domElement.removeEventListener('pointermove', onPointerMove);
			renderer.domElement.removeEventListener('pointerleave', onPointerLeave);
			geometry.dispose();
			starMaterial.dispose();
			haloMaterial.dispose();
			renderer.dispose();
			renderer.domElement.remove();
		};
	});

	onDestroy(() => {
		if (cleanup) cleanup();
	});
</script>

<div class="relative w-full" style="height: {height}; user-select: none;">
	<div bind:this={container} class="absolute inset-0"></div>

	{#if loading}
		<div
			class="absolute inset-0 grid place-items-center pointer-events-none"
			style="color: var(--ink-dim);"
		>
			<div class="text-sm flex items-center gap-2">
				<span
					class="inline-block w-1.5 h-1.5 rounded-full"
					style="background: var(--brand); box-shadow: 0 0 12px var(--brand); animation: pulse-slow 1.5s ease-in-out infinite;"
				></span>
				warming up the cloud…
			</div>
		</div>
	{/if}

	{#if error}
		<div class="absolute inset-0 grid place-items-center text-xs" style="color: #fca5a5;">
			{error}
		</div>
	{/if}

	{#if !loading && !error}
		<!-- Hover readout / hint -->
		<div
			class="absolute top-3 left-1/2 -translate-x-1/2 px-3 py-1.5 rounded-full text-xs mono pointer-events-none transition"
			style="
				background: rgba(10, 12, 16, 0.78);
				border: 1px solid var(--border);
				color: {hoveredTitle ? 'var(--ink)' : 'var(--ink-faint)'};
				backdrop-filter: blur(6px);
				opacity: {hoveredTitle ? 1 : 0.6};
				white-space: nowrap;
				max-width: 80vw;
				overflow: hidden;
				text-overflow: ellipsis;
			"
		>
			{#if hoveredTitle}
				<span style="color: {hoveredGenre ? 'var(--ink)' : 'inherit'};">
					{hoveredTitle}
				</span>
				{#if hoveredGenre}
					<span style="color: var(--ink-faint); margin-left: 0.4rem;">· {hoveredGenre}</span>
				{/if}
			{:else}
				✦  drift through the latent space  ✦  drag to look around
			{/if}
		</div>

		<!-- Bottom edge fade so the cloud melts into the page -->
		<div
			class="absolute inset-x-0 bottom-0 h-32 pointer-events-none"
			style="background: linear-gradient(180deg, transparent, var(--bg) 90%);"
		></div>
	{/if}
</div>
