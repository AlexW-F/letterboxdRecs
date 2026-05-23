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
		// Per-star seed (0..1) for twinkle phase offset.
		const seeds = new Float32Array(raw.length);

		// Slight per-point size variation by popularity gives the cloud a
		// near-field / far-field star feel; popular films are visibly larger.
		const popMax = Math.max(...data.popularity);
		for (let i = 0; i < raw.length; i++) {
			positions[3 * i] = (raw[i][0] - cx) * scale;
			positions[3 * i + 1] = (raw[i][1] - cy) * scale;
			positions[3 * i + 2] = (raw[i][2] - cz) * scale;
			const hex = data.genre_colors[data.genres[i]] ?? '#cccccc';
			c.set(hex);
			// Saturate colors so they read against the dark scene.
			const hsl = { h: 0, s: 0, l: 0 };
			c.getHSL(hsl);
			c.setHSL(hsl.h, Math.min(1, hsl.s * 1.4 + 0.1), Math.max(0.55, Math.min(0.75, hsl.l)));
			colors[3 * i] = c.r;
			colors[3 * i + 1] = c.g;
			colors[3 * i + 2] = c.b;
			// Size scales loosely with log(popularity); range ~0.55–2.6.
			const popN = Math.log1p(data.popularity[i]) / Math.log1p(popMax);
			sizes[i] = 0.55 + popN * 2.05;
			seeds[i] = Math.random();
		}

		const geometry = new THREE.BufferGeometry();
		geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
		geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
		geometry.setAttribute('aSize', new THREE.BufferAttribute(sizes, 1));
		geometry.setAttribute('aSeed', new THREE.BufferAttribute(seeds, 1));

		// ---------------------------------------------------------------
		// Galaxy-star shader — bright core + wide exponential halo, with
		// per-star twinkle, depth fade, and additive blending so dense
		// regions naturally bloom brighter.
		// ---------------------------------------------------------------
		const starMaterial = new THREE.ShaderMaterial({
			uniforms: {
				uPixelRatio: { value: Math.min(window.devicePixelRatio, 2) },
				uSize: { value: 90.0 },
				uTime: { value: 0 }
			},
			vertexShader: /* glsl */ `
				attribute float aSize;
				attribute float aSeed;
				varying vec3 vColor;
				varying float vDist;
				varying float vTwinkle;
				uniform float uPixelRatio;
				uniform float uSize;
				uniform float uTime;
				void main() {
					vColor = color;
					vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
					vDist = -mvPosition.z;
					// Smoothly modulate per-star brightness 0.7..1.15 with a low-
					// frequency sinusoid offset by the star's own seed.
					vTwinkle = 0.85 + 0.30 * sin(uTime * 1.3 + aSeed * 18.0);
					gl_Position = projectionMatrix * mvPosition;
					// Bigger close stars — divide by depth, clamp by GPU limit.
					gl_PointSize = uSize * aSize * uPixelRatio / max(2.0, -mvPosition.z);
				}
			`,
			fragmentShader: /* glsl */ `
				varying vec3 vColor;
				varying float vDist;
				varying float vTwinkle;
				void main() {
					// 0 at center, 1 at the edge of the point square.
					vec2 d = gl_PointCoord - 0.5;
					float r = length(d) * 2.0;
					if (r > 1.0) discard;

					// Wide soft halo: rapid exponential falloff. This is what
					// gives the galaxy-glow when many overlap.
					float halo = exp(-r * 3.4);

					// Tight bright core: highly peaked near r=0.
					float core = pow(1.0 - r, 7.0);

					// Whiten the core a touch — real stars look hotter at center.
					vec3 col = mix(vColor, vec3(1.0), core * 0.55);
					float intensity = (halo * 0.75 + core * 2.2) * vTwinkle;

					// Depth fade so distant stars dissolve into the fog.
					float depthFade = 1.0 - smoothstep(22.0, 72.0, vDist);
					intensity *= 0.45 + 0.55 * depthFade;

					gl_FragColor = vec4(col * intensity, intensity);
				}
			`,
			blending: THREE.AdditiveBlending,
			depthWrite: false,
			transparent: true,
			vertexColors: true
		});

		const stars = new THREE.Points(geometry, starMaterial);
		scene.add(stars);

		// Second pass — much bigger, softer points for the diffuse glow
		// around dense regions. Together with additive blending this is
		// what turns "scatter plot" into "galaxy".
		const glowMaterial = new THREE.ShaderMaterial({
			uniforms: {
				uPixelRatio: { value: Math.min(window.devicePixelRatio, 2) },
				uSize: { value: 260.0 },
				uTime: { value: 0 }
			},
			vertexShader: /* glsl */ `
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
					gl_PointSize = uSize * aSize * uPixelRatio / max(2.0, -mvPosition.z);
				}
			`,
			fragmentShader: /* glsl */ `
				varying vec3 vColor;
				varying float vDist;
				void main() {
					vec2 d = gl_PointCoord - 0.5;
					float r = length(d) * 2.0;
					if (r > 1.0) discard;
					// Pure exponential, very soft — invisible at the edges, faintly
					// tinted near the center. Many overlapping copies blend into
					// dust lanes.
					float glow = exp(-r * 5.0) * 0.18;
					float depthFade = 1.0 - smoothstep(25.0, 80.0, vDist);
					glow *= 0.4 + 0.6 * depthFade;
					gl_FragColor = vec4(vColor * glow, glow);
				}
			`,
			blending: THREE.AdditiveBlending,
			depthWrite: false,
			transparent: true,
			vertexColors: true
		});

		const glow = new THREE.Points(geometry, glowMaterial);
		scene.add(glow);

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
			glowMaterial.uniforms.uTime.value = t;

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
			glow.rotation.y = t * 0.012;

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
			glowMaterial.dispose();
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
