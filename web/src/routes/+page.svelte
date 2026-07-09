<script lang="ts">
	import { goto } from '$app/navigation';
	import { Users as UsersIcon, Loader2 } from 'lucide-svelte';
	import MovieSpaceHero from '$lib/components/MovieSpaceHero.svelte';
	import { createGroup } from '$lib/api';

	let creatingGroup = $state(false);
	let error = $state<string | null>(null);

	// Device-aware hero density. The hero builds an O(n²) k-NN web at mount,
	// which freezes low-power phones at ~5000 points. Scale the cloud down on
	// small / low-core / low-memory devices, up to ~3500 on capable desktops.
	// SSR-safe: defaults to a conservative count when there's no window.
	function computeHeroPoints(): number {
		if (typeof window === 'undefined') return 2000;
		const w = window.innerWidth;
		const cores = navigator.hardwareConcurrency ?? 4;
		// deviceMemory is non-standard (Chromium-only); treat absent as mid-range.
		const mem = (navigator as Navigator & { deviceMemory?: number }).deviceMemory ?? 4;

		if (w < 640 || cores <= 4 || mem <= 4) return 2500;
		if (w < 1024 || cores <= 6 || mem <= 6) return 3000;
		return 3500;
	}

	const heroPoints = computeHeroPoints();

	async function createShareableGroup() {
		error = null;
		creatingGroup = true;
		try {
			const g = await createGroup();
			goto(`/group/${g.group_id}/join`);
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			creatingGroup = false;
		}
	}

</script>

<!--
  The cloud is the whole show. Full-viewport Three.js canvas with drag
  enabled; the copy lives in a poster-style billing block anchored to the
  bottom of the first viewport. Everything non-interactive is
  pointer-events: none so dragging works even over the headline.
-->

<div
	class="fixed inset-0 -z-10"
	style="pointer-events: auto; isolation: isolate;"
>
	<MovieSpaceHero
		height="100vh"
		points={heroPoints}
		enableDrag={true}
		parallaxStrength={0.35}
	/>
</div>

<!-- Progressive frost under the billing block: a low backdrop blur whose
     strength fades in via a mask, so there's no hard edge where the filter
     starts. Sits between the canvas and the text — it only ever blurs the
     cloud, never the type. -->
<div
	class="fixed inset-x-0 bottom-0 -z-[6] pointer-events-none"
	style="
		height: 48vh;
		backdrop-filter: blur(7px) saturate(115%);
		-webkit-backdrop-filter: blur(7px) saturate(115%);
		mask-image: linear-gradient(180deg, transparent 0%, rgba(0, 0, 0, 0.5) 38%, #000 78%);
		-webkit-mask-image: linear-gradient(180deg, transparent 0%, rgba(0, 0, 0, 0.5) 38%, #000 78%);
	"
></div>

<!-- Bottom scrim so the billing block reads cleanly against the cloud -->
<div
	class="fixed inset-x-0 bottom-0 -z-[5] pointer-events-none"
	style="height: 55vh; background: linear-gradient(180deg, transparent 0%, rgba(23, 22, 19, 0.5) 55%, rgba(20, 19, 16, 0.82) 100%);"
></div>

<section class="relative" style="z-index: 1; pointer-events: none;">
	<!-- Poster billing block: headline bottom-left, actions bottom-right. -->
	<div
		class="anim-fade-up flex flex-col justify-end"
		style="min-height: calc(100vh - 190px); pointer-events: none;"
	>
		<div class="flex flex-wrap items-end justify-between gap-x-10 gap-y-8 pb-4">
			<div class="max-w-2xl">
				<div
					class="board neon-green"
					style="font-size: 0.72rem; letter-spacing: 0.42em;"
				>
					— open all night —
				</div>
				<!-- No per-line scrims here: at Oswald's tight leading the cloned
				     background strips overlap the previous line's glyphs and paint
				     over them, ghosting the headline. The scrim gradient + a soft
				     shadow carry readability instead. -->
				<h1
					class="display-xl mt-4"
					style="text-shadow: 0 2px 26px rgba(10, 9, 7, 0.9), 0 0 60px rgba(10, 9, 7, 0.7);"
				>
					Movies your whole<br />
					group <span class="neon-amber">will love</span>
				</h1>
				<p
					class="mt-4 max-w-xl"
					style="color: var(--ink-muted); font-size: 1.02rem; line-height: 1.75;"
				>
					Everybody tosses in their Letterboxd, the projector does the math, and you argue the
					shortlist down to one film. Doors at eight.
				</p>
			</div>

			<div
				class="flex flex-col items-start sm:items-end gap-3 pb-1"
				style="pointer-events: auto; max-width: 23rem;"
			>
				<button
					class="btn btn-primary"
					style="font-size: 0.9rem; padding: 0.75rem 1.6rem;"
					onclick={createShareableGroup}
					disabled={creatingGroup}
				>
					{#if creatingGroup}
						<Loader2 size={16} class="animate-spin" />
						lighting it up…
					{:else}
						<UsersIcon size={16} />
						Put tonight on the marquee
					{/if}
				</button>
				<div class="flex items-center gap-2">
					<button
						class="btn btn-ghost text-sm"
						style="color: var(--ink); opacity: 1;"
						onclick={() => goto('/me')}
					>
						just me →
					</button>
					<a href="/explore" class="btn btn-ghost text-sm" style="color: var(--ink-muted);">
						wander the cloud →
					</a>
				</div>
				<p class="text-xs mono sm:text-right" style="color: var(--ink-faint);">
					{heroPoints.toLocaleString()} films floating up there · every light is real ·
					<span style="color: var(--ink-dim);">drag to look around</span>
				</p>
				{#if error}
					<p class="text-xs" style="color: var(--rust);">{error}</p>
				{/if}
			</div>
		</div>
	</div>
</section>
