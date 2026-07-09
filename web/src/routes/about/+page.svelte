<script lang="ts">
	import { onMount } from 'svelte';
	import { Database, Layers, Boxes, Shield, Cpu } from 'lucide-svelte';
	import Stat from '$lib/components/Stat.svelte';
	import { getHealth, type Health } from '$lib/api';

	let health = $state<Health | null>(null);
	let error = $state<string | null>(null);

	onMount(() => {
		getHealth()
			.then((h) => (health = h))
			.catch((e) => (error = e instanceof Error ? e.message : String(e)));
	});

	// Plain-language cards, relocated from the landing page so the cloud
	// gets the whole stage there. The technical stages below stay technical.
	const whatItDoes: { title: string; body: string; tint: string; href?: string }[] = [
		{
			title: 'Six ways to agree',
			body: 'Fair averages, least-misery, or fuse the whole group into one taste and let the projector decide. Pick the politics that fit your friends.',
			tint: 'chip-violet'
		},
		{
			title: 'Knows the deep cuts',
			body: '1,084 hand-curated tags plus plot embeddings under the hood — so the shortlist reaches past the same twenty films everyone recommends.',
			tint: 'chip-brand'
		},
		{
			title: 'Your taste, in 3D',
			body: 'The lights on the landing page are real films. In the explore room you drift through them with your own ratings lit up around you.',
			tint: 'chip-rose',
			href: '/explore'
		}
	];

	const stages = [
		{
			title: 'Candidate generation',
			body: 'Union of top-K from two trained models on ml-32m (200,948 users × 84,432 items). Surprise SVD (64 factors, 20 epochs, scale 0.5–5.0) for explicit ratings; implicit ALS for positives (≥3.5) with Hu/Koren confidence weighting.'
		},
		{
			title: 'Re-ranking',
			body: 'Weighted blend of SVD score + ALS score − popularity penalty − MMR diversity penalty + genre-overlap bonus + content cosine. Per-mode weights at /modes.'
		},
		{
			title: 'Content scoring',
			body: 'Tag Genome 2021 (1,084 curated tags × 9,734 films, deep-learning-fitted relevance) as primary. TMDB plot embeddings via all-MiniLM-L6-v2 (384-d) at weight 0.3 for thematic adjacency. Director one-hots from IMDb opt-in via CONTENT_FEATURES_EXTRA.'
		},
		{
			title: 'Group aggregation',
			body: 'Six strategies, each operating on real per-member predictions (not zero-filled top-N): average, least_misery, most_pleasure, consensus, hybrid, group_taste_vector. The last one fuses all members into one super-user and re-ranks against the merged signal.'
		}
	];
</script>

<svelte:head>
	<title>About · movienight</title>
</svelte:head>

<section class="space-y-8 max-w-4xl anim-fade-up">
	<header>
		<span class="chip">about · methodology</span>
		<h1 class="display-md mt-2">
			How the picks <span class="text-gradient">happen</span>.
		</h1>
		<p class="text-sm" style="color: var(--ink-muted);">
			A four-stage hybrid recommender — collaborative filtering meets content semantics, with
			group-aware aggregation as a first-class step.
		</p>
	</header>

	<div class="grid grid-cols-1 md:grid-cols-3 gap-3">
		{#each whatItDoes as item, i (item.title)}
			<svelte:element
				this={item.href ? 'a' : 'div'}
				href={item.href}
				class="surface p-4 card-hover anim-fade-up block {i === 0 ? 'tilt-a' : i === 2 ? 'tilt-b' : ''}"
				style="animation-delay: {i * 50}ms;"
			>
				<span class="chip {item.tint}">{item.title}</span>
				<p class="text-sm mt-3" style="color: var(--ink-muted); line-height: 1.5;">{item.body}</p>
				{#if item.href}
					<p class="text-xs mt-2 board" style="color: var(--brand); letter-spacing: 0.14em;">step inside →</p>
				{/if}
			</svelte:element>
		{/each}
	</div>

	{#if health}
		<div class="grid grid-cols-2 md:grid-cols-4 gap-3">
			<Stat label="catalog" value={health.catalog_size.toLocaleString()} hint="ml-32m films" />
			<Stat label="SVD" value={health.svd_loaded ? 'loaded' : '—'} hint="explicit ratings" />
			<Stat label="ALS" value={health.als_loaded ? 'loaded' : '—'} hint="implicit feedback" />
			<Stat label="content" value={health.content_loaded ? 'loaded' : '—'} hint="genome + overviews" />
		</div>
	{:else if error}
		<p class="text-sm" style="color: var(--rust);">{error}</p>
	{/if}

	<div class="grid grid-cols-1 md:grid-cols-2 gap-3">
		{#each stages as s, i (s.title)}
			<div
				class="surface p-5 card-hover anim-fade-up"
				style="animation-delay: {i * 40}ms;"
			>
				<div class="flex items-center gap-2 mb-2" style="color: var(--brand);">
					{#if i === 0}<Database size={16} />{/if}
					{#if i === 1}<Layers size={16} />{/if}
					{#if i === 2}<Boxes size={16} />{/if}
					{#if i === 3}<Cpu size={16} />{/if}
					<h3 class="text-base font-medium" style="color: var(--ink);">{s.title}</h3>
				</div>
				<p class="text-sm" style="color: var(--ink-muted); line-height: 1.55;">{s.body}</p>
			</div>
		{/each}
	</div>

	{#if health}
		<div class="surface p-5">
			<h2 class="text-base font-medium mb-3 flex items-center gap-2">
				<Cpu size={16} style="color: var(--violet);" />
				Live model state
			</h2>
			<dl class="grid grid-cols-3 gap-x-4 gap-y-2 text-sm">
				<dt style="color: var(--ink-faint);">status</dt>
				<dd class="col-span-2">{health.status}</dd>

				<dt style="color: var(--ink-faint);">catalog</dt>
				<dd class="col-span-2">{health.catalog_size.toLocaleString()} films</dd>

				<dt style="color: var(--ink-faint);">SVD</dt>
				<dd class="col-span-2">{health.svd_loaded ? 'loaded' : '—'}</dd>

				<dt style="color: var(--ink-faint);">ALS</dt>
				<dd class="col-span-2">{health.als_loaded ? 'loaded' : '—'}</dd>

				<dt style="color: var(--ink-faint);">content</dt>
				<dd class="col-span-2">{health.content_loaded ? 'loaded' : '—'}</dd>

				<dt style="color: var(--ink-faint);">3D viz</dt>
				<dd class="col-span-2">{health.movie_space_loaded ? 'loaded' : 'static fallback'}</dd>

				<dt style="color: var(--ink-faint);">model files</dt>
				<dd class="col-span-2 mono text-xs" style="color: var(--ink-muted);">{health.model_name}</dd>
			</dl>
		</div>
	{/if}

	<div class="surface p-5">
		<div class="flex items-center gap-2 mb-3">
			<Shield size={16} style="color: var(--brand);" />
			<h2 class="text-base font-medium">Privacy</h2>
		</div>
		<p class="text-sm" style="color: var(--ink-muted);">
			Uploaded ratings are content-hashed (SHA-256 of sorted CSV lines) and cached server-side
			against that hash. No session lifecycle, no PII at rest beyond the rating/movieId pairs.
			Re-uploading the same export is a cache hit. Clearing the
			<code class="mono">.api_cache/</code> directory wipes everything.
		</p>
	</div>
</section>
