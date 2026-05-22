<script lang="ts">
	import { getHealth, type Health } from '$lib/api';

	let health = $state<Health | null>(null);
	let error = $state<string | null>(null);

	$effect(() => {
		getHealth()
			.then((h) => (health = h))
			.catch((e) => (error = e instanceof Error ? e.message : String(e)));
	});
</script>

<section class="space-y-6 max-w-3xl">
	<h1 class="text-2xl font-semibold tracking-tight">About</h1>

	<div class="rounded-lg bg-white/5 border border-white/10 p-4 space-y-2 text-sm">
		<h2 class="text-base font-medium">Pipeline</h2>
		<ol class="list-decimal list-inside space-y-1 text-white/70">
			<li>
				<strong>Candidate generation</strong> — union of top-K from two trained models on the
				MovieLens 32M catalog (200,948 users × 84,432 items).
				<ul class="list-disc list-inside ml-4 mt-1 text-white/50 text-xs">
					<li>
						Surprise <code>SVD</code> on explicit ratings — 64 latent factors, 20 epochs, rating
						scale 0.5–5.0.
					</li>
					<li>
						<code>implicit</code> <code>AlternatingLeastSquares</code> on positive ratings (≥ 3.5)
						with Hu/Koren confidence weighting.
					</li>
				</ul>
			</li>
			<li>
				<strong>Re-ranking</strong> — weighted blend of SVD, ALS, popularity penalty, MMR diversity
				penalty, implicit genre-overlap bonus, and TF-IDF content cosine. Per-mode weights
				expressed as <code>/modes</code>.
			</li>
			<li>
				<strong>Content scoring</strong> — TF-IDF over MovieLens-32M's 2M user-generated tags +
				genres (20k vocab). Each user's taste vector is a (rating − mean)-weighted average of
				their rated movies' rows. Group taste vector fuses members.
			</li>
			<li>
				<strong>Group aggregation</strong> — 6 strategies, including <code>group_taste_vector</code>
				which fuses all members into one super-user and re-ranks against the fused signal.
			</li>
		</ol>
	</div>

	<div class="rounded-lg bg-white/5 border border-white/10 p-4 space-y-2 text-sm">
		<h2 class="text-base font-medium">Models loaded</h2>
		{#if error}
			<p class="text-red-400">{error}</p>
		{:else if health}
			<dl class="grid grid-cols-3 gap-x-4 gap-y-1 text-xs">
				<dt class="text-white/40">status</dt>
				<dd class="col-span-2 text-white">{health.status}</dd>

				<dt class="text-white/40">catalog</dt>
				<dd class="col-span-2 text-white">{health.catalog_size.toLocaleString()} films</dd>

				<dt class="text-white/40">SVD</dt>
				<dd class="col-span-2 text-white">{health.svd_loaded ? 'loaded' : '—'}</dd>

				<dt class="text-white/40">ALS</dt>
				<dd class="col-span-2 text-white">{health.als_loaded ? 'loaded' : '—'}</dd>

				<dt class="text-white/40">content</dt>
				<dd class="col-span-2 text-white">{health.content_loaded ? 'loaded' : '—'}</dd>

				<dt class="text-white/40">model files</dt>
				<dd class="col-span-2 text-white">{health.model_name}</dd>

				<dt class="text-white/40">cache</dt>
				<dd class="col-span-2 text-white/60 text-[11px]">{health.cache_dir}</dd>
			</dl>
		{:else}
			<p class="text-white/40">connecting…</p>
		{/if}
	</div>

	<div class="rounded-lg bg-white/5 border border-white/10 p-4 space-y-2 text-sm">
		<h2 class="text-base font-medium">Privacy</h2>
		<p class="text-white/60">
			Uploaded ratings are content-hashed (SHA-256 of sorted CSV lines) and cached server-side
			against that hash. There's no session lifecycle and no PII at rest beyond the rating/movieId
			pairs. Re-uploading the same export is a cache hit. Clearing the cache directory
			(<code>.api_cache/</code>) wipes everything.
		</p>
	</div>
</section>
