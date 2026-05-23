<script lang="ts">
	// Embeds the 3D UMAP projection of ml-32m's ALS item-factor space.
	// The HTML file at /movie_space.html is fully self-contained (plotly via CDN);
	// rebuild it with `scripts/build_movie_space_viz.py` and re-copy into
	// web/static/movie_space.html to refresh.
</script>

<section class="space-y-4">
	<header>
		<h1 class="text-2xl font-semibold tracking-tight">Movie space</h1>
		<p class="text-sm text-white/50">
			Each dot is a film, positioned by its ALS latent factors projected to 3D via UMAP. Color
			encodes primary genre. Alex's rated films are bigger, colored by her rating; her folded-in
			taste vector is the black diamond. Drag to rotate, scroll to zoom, hover for titles.
		</p>
		<p class="text-xs text-white/40">
			Built from <code>models/als_full.pkl</code> via <code>scripts/build_movie_space_viz.py</code>.
			Static snapshot — re-run that script to refresh with a different user.
		</p>
	</header>

	<div class="rounded-lg border border-white/10 bg-black/30 overflow-hidden">
		<iframe
			src="/movie_space.html"
			title="3D movie space"
			class="w-full"
			style="height: 80vh; border: 0; background: #0b0d11;"
		></iframe>
	</div>

	<p class="text-xs text-white/40">
		If the embed feels cramped, <a href="/movie_space.html" class="text-sky-300 hover:text-sky-200">open it in its own tab</a>.
	</p>
</section>
