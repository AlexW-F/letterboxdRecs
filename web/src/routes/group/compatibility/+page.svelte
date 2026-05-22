<script lang="ts">
	import { analyzeGroup, type GroupAnalysis } from '$lib/api';
	import { loadMembers } from '$lib/store';

	const members = $state(loadMembers());
	let busy = $state(true);
	let error = $state<string | null>(null);
	let report = $state<GroupAnalysis | null>(null);

	$effect(() => {
		if (members.length < 2) {
			busy = false;
			return;
		}
		analyzeGroup({
			hashes: members.map((m) => m.hash),
			member_names: members.map((m) => m.name),
			top_overlap: 10
		})
			.then((r) => (report = r))
			.catch((e) => (error = e instanceof Error ? e.message : String(e)))
			.finally(() => (busy = false));
	});

	function sigClass(v: number | null): string {
		if (v === null) return 'text-white/30';
		if (v >= 0.3) return 'text-emerald-300';
		if (v <= -0.3) return 'text-red-300';
		return 'text-white/60';
	}
	function sigLabel(v: number | null): string {
		if (v === null) return 'n/a';
		return v.toFixed(2);
	}
</script>

<section class="space-y-6">
	<header class="flex items-center justify-between">
		<div>
			<h1 class="text-2xl font-semibold tracking-tight">Compatibility report</h1>
			<p class="text-sm text-white/50">How your group's tastes align before the rec list.</p>
		</div>
		<a href="/group/recommendations" class="text-sm text-sky-300 hover:text-sky-200"
			>back to recs →</a
		>
	</header>

	{#if members.length < 2}
		<div class="rounded-lg bg-red-500/10 border border-red-400/30 p-4 text-sm">
			Need at least 2 members. <a href="/" class="underline">Add some →</a>
		</div>
	{:else if busy}
		<p class="text-white/40">computing…</p>
	{:else if error}
		<p class="text-red-400">{error}</p>
	{:else if report}
		<div class="space-y-6">
			<div>
				<h2 class="text-lg font-medium mb-2">Pairwise similarity</h2>
				<table class="w-full text-sm">
					<thead class="text-white/40 text-xs">
						<tr class="text-left">
							<th class="py-2">pair</th>
							<th>shared films</th>
							<th>pearson on shared</th>
							<th>cosine on taste vec</th>
						</tr>
					</thead>
					<tbody>
						{#each report.pairwise as p (p.pair.join('+'))}
							<tr class="border-t border-white/5">
								<td class="py-2 font-medium">{p.pair[0]} ↔ {p.pair[1]}</td>
								<td class="text-white/60">{p.n_shared_movies}</td>
								<td class={sigClass(p.pearson_on_shared) + ' tabular-nums'}>
									{sigLabel(p.pearson_on_shared)}
								</td>
								<td class={sigClass(p.cosine_on_taste) + ' tabular-nums'}>
									{sigLabel(p.cosine_on_taste)}
								</td>
							</tr>
						{/each}
					</tbody>
				</table>
				<p class="text-xs text-white/30 mt-1">
					Pearson = correlation of ratings on films both have seen. Cosine = similarity of TF-IDF
					taste vectors across the whole catalog. Green ≥ 0.3, red ≤ -0.3.
				</p>
			</div>

			<div class="grid grid-cols-1 md:grid-cols-2 gap-6">
				<div>
					<h2 class="text-lg font-medium mb-2">Consensus picks</h2>
					<p class="text-xs text-white/40 mb-2">High mean rating + low std across all members.</p>
					<ul class="space-y-1.5 text-sm">
						{#each report.consensus_movies as m (m.movie_id)}
							<li class="bg-white/[0.03] border border-white/5 rounded px-3 py-2">
								<div class="font-medium">{m.title}</div>
								<div class="text-xs text-white/50">
									avg {m.mean.toFixed(2)}★ · std {m.std.toFixed(2)}
								</div>
								<div class="text-[11px] text-white/40">
									{#each Object.entries(m.member_ratings) as [n, r], i (n)}{#if i > 0}, {/if}{n}
										{r}★{/each}
								</div>
							</li>
						{/each}
					</ul>
				</div>

				<div>
					<h2 class="text-lg font-medium mb-2">Disagreement picks</h2>
					<p class="text-xs text-white/40 mb-2">
						High std across members — these are the films you'd fight about.
					</p>
					<ul class="space-y-1.5 text-sm">
						{#each report.disagreement_movies as m (m.movie_id)}
							<li class="bg-white/[0.03] border border-white/5 rounded px-3 py-2">
								<div class="font-medium">{m.title}</div>
								<div class="text-xs text-white/50">
									avg {m.mean.toFixed(2)}★ · std {m.std.toFixed(2)}
								</div>
								<div class="text-[11px] text-white/40">
									{#each Object.entries(m.member_ratings) as [n, r], i (n)}{#if i > 0}, {/if}{n}
										{r}★{/each}
								</div>
							</li>
						{/each}
					</ul>
				</div>
			</div>
		</div>
	{/if}
</section>
