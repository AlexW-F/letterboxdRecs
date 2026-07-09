<script lang="ts">
	import { goto } from '$app/navigation';
	import {
		Sparkles,
		Loader2,
		RefreshCw,
		ArrowLeft,
		BarChart3,
		Users as UsersIcon,
		Eye
	} from 'lucide-svelte';
	import ModeSelector from '$lib/components/ModeSelector.svelte';
	import StrategySelector from '$lib/components/StrategySelector.svelte';
	import RecommendationCard from '$lib/components/RecommendationCard.svelte';
	import Avatar from '$lib/components/Avatar.svelte';
	import Stat from '$lib/components/Stat.svelte';
	import Shimmer from '$lib/components/Shimmer.svelte';
	import { page } from '$app/state';
	import {
		getModes,
		getStrategies,
		recommendGroup,
		recommendGroupDisagreement,
		groupWatchlistOverlap,
		getGroup,
		castGroupVote,
		type GroupRecResponse,
		type GroupState,
		type Mode,
		type Strategy,
		type WatchlistOverlapResponse
	} from '$lib/api';
	import { loadMembers } from '$lib/store';

	// Shared-group mode: ?group=ID pulls members from server state.
	// Otherwise fall back to the local browser members (CSV uploads on /).
	const groupId = $derived(page.url.searchParams.get('group'));
	let serverGroup = $state<GroupState | null>(null);
	const localMembers = $state(loadMembers());

	const members = $derived(
		serverGroup
			? serverGroup.members.map((m) => ({
					name: m.name,
					hash: m.hash,
					upload: {
						hash: m.hash,
						n_ratings_in: 0,
						n_ratings_mapped: m.n_ratings_mapped,
						n_with_tmdb: 0,
						n_watchlist: m.n_watchlist
					}
				}))
			: localMembers
	);

	$effect(() => {
		if (groupId) {
			getGroup(groupId)
				.then((g) => (serverGroup = g))
				.catch((e) => (error = e instanceof Error ? e.message : String(e)));
		}
	});

	// "I am" voter identity — persisted per-group in localStorage.
	let voterName = $state<string>('');
	const voterKey = $derived(groupId ? `lbrecs_voter::${groupId}` : '');
	$effect(() => {
		if (!voterKey) return;
		const saved = localStorage.getItem(voterKey);
		if (saved) voterName = saved;
	});
	$effect(() => {
		if (voterKey && voterName) localStorage.setItem(voterKey, voterName);
	});

	// Default voter once the group loads (if nothing saved yet): prefer the
	// member whose upload lives on this device — defaulting to members[0]
	// would cast votes as whoever joined first, not the person holding the phone.
	$effect(() => {
		if (!voterName && serverGroup && serverGroup.members.length > 0) {
			const localHashes = new Set(localMembers.map((m) => m.hash));
			const mine = serverGroup.members.find((m) => localHashes.has(m.hash));
			voterName = (mine ?? serverGroup.members[0]).name;
		}
	});

	// Winner = highest (ups - vetos) across all voted-on films currently shown.
	// Requires at least one up-vote to surface.
	const winner = $derived.by(() => {
		if (!serverGroup || !topResult) return null;
		const titleOf = new Map(topResult.recommendations.map((r) => [r.movie_id, r.title]));
		// Also include argue-tab picks if loaded
		if (argueResult) {
			for (const r of argueResult.recommendations) titleOf.set(r.movie_id, r.title);
		}
		let best: { movie_id: string; title: string; up: number; veto: number; net: number } | null = null;
		for (const [mid, ballots] of Object.entries(serverGroup.votes)) {
			if (!titleOf.has(mid)) continue;
			let up = 0, veto = 0;
			for (const v of Object.values(ballots)) {
				if (v === 'up') up++;
				else if (v === 'veto') veto++;
			}
			if (up === 0) continue;
			const net = up - veto;
			if (!best || net > best.net) {
				best = { movie_id: mid, title: titleOf.get(mid)!, up, veto, net };
			}
		}
		return best;
	});

	async function handleVote(movieId: string, vote: 'up' | 'veto' | 'clear') {
		if (!groupId || !voterName) return;
		try {
			const updated = await castGroupVote(groupId, {
				member_name: voterName,
				movie_id: movieId,
				vote
			});
			serverGroup = updated;
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		}
	}

	// Light polling so other members' votes show up without a manual refresh
	$effect(() => {
		if (!groupId) return;
		const id = setInterval(() => {
			getGroup(groupId).then((g) => (serverGroup = g)).catch(() => {});
		}, 6000);
		return () => clearInterval(id);
	});

	let modes = $state<Mode[]>([]);
	let strategies = $state<Strategy[]>([]);
	let mode = $state('balanced');
	let strategy = $state('group_taste_vector');
	let topN = $state(12);

	let busy = $state(false);
	let error = $state<string | null>(null);
	// Params of the last computed run — used to flag stale results when the
	// user changes strategy/mode but hasn't hit recompute yet.
	let lastRunKey = $state<string | null>(null);
	const paramsKey = $derived(`${strategy}|${mode}|${topN}`);
	const stale = $derived(lastRunKey !== null && lastRunKey !== paramsKey && !busy);
	let topResult = $state<GroupRecResponse | null>(null);
	let argueResult = $state<GroupRecResponse | null>(null);
	let seenResult = $state<GroupRecResponse | null>(null);
	let watchlistResult = $state<WatchlistOverlapResponse | null>(null);
	let watchlistExpanded = $state(false);
	let view = $state<'top' | 'argue' | 'seen'>('top');

	const activeResult = $derived(
		view === 'argue' ? argueResult : view === 'seen' ? seenResult : topResult
	);
	const membersWithWatchlist = $derived(members.filter((m) => (m.upload?.n_watchlist ?? 0) > 0));

	$effect(() => {
		Promise.all([getModes(), getStrategies()])
			.then(([m, s]) => {
				modes = m.modes;
				strategies = s.strategies;
			})
			// API down: run() surfaces the real error; selectors just stay empty.
			.catch(() => {});
	});

	async function run() {
		error = null;
		if (members.length < 2) {
			error = 'Add at least 2 group members on the landing page first.';
			return;
		}
		busy = true;
		lastRunKey = paramsKey;
		const payload = {
			hashes: members.map((m) => m.hash),
			member_names: members.map((m) => m.name),
			strategy,
			mode,
			top_n: topN,
			exclude_rated: true,
			exclude_watched: true
		};
		try {
			// Fetch top picks (per-member exclude), disagreement, "nobody's seen"
			// (strict union exclude — no member can have rated or watched), and
			// watchlist overlap in parallel — toggling between tabs is then instant.
			const nobodysSeenPayload = { ...payload, exclude_seen_by_any: true };
			const [top, argue, seen, watchlist] = await Promise.all([
				recommendGroup(payload),
				recommendGroupDisagreement(payload).catch(() => null),
				recommendGroup(nobodysSeenPayload).catch(() => null),
				membersWithWatchlist.length >= 2
					? groupWatchlistOverlap({
							hashes: members.map((m) => m.hash),
							member_names: members.map((m) => m.name),
							min_members: 2,
							top_n: 30
						}).catch(() => null)
					: Promise.resolve(null)
			]);
			topResult = top;
			argueResult = argue;
			seenResult = seen;
			watchlistResult = watchlist;
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			busy = false;
		}
	}

	$effect(() => {
		// Don't auto-run with stale local members while a server group is loading
		const ready = !groupId || serverGroup;
		if (ready && members.length >= 2 && !topResult && !busy) run();
	});

	const totalRated = $derived(
		members.reduce((acc, m) => acc + (m.upload?.n_ratings_mapped ?? 0), 0)
	);
</script>

<svelte:head>
	<title>Group picks · movienight</title>
</svelte:head>

<section class="space-y-6 anim-fade-up">
	<header class="flex items-end justify-between flex-wrap gap-3">
		<div>
			<span class="chip chip-violet">
				<UsersIcon size={11} />
				Group · {members.length} members · {strategy.replace(/_/g, ' ')}
			</span>
			<h1 class="display-md mt-2">
				Picks <span class="text-gradient">your group</span> will love.
			</h1>
			<div class="mt-2 flex flex-wrap items-center gap-2">
				{#each members as m (m.hash)}
					<div class="flex items-center gap-1.5 chip" style="padding: 0.2rem 0.5rem 0.2rem 0.2rem;">
						<Avatar name={m.name} size={20} />
						<span class="text-xs normal-case" style="color: var(--ink-muted); letter-spacing: 0;">
							{m.name}
						</span>
					</div>
				{/each}
			</div>
		</div>
		<div class="flex gap-2 text-sm">
			<a
				href={groupId ? `/group/compatibility?group=${encodeURIComponent(groupId)}` : '/group/compatibility'}
				class="btn btn-ghost btn-pill text-xs"
			>
				<BarChart3 size={13} />
				compatibility
			</a>
			<button
				class="btn btn-ghost btn-pill text-xs"
				onclick={() => goto(groupId ? `/group/${groupId}/join` : '/')}
			>
				<ArrowLeft size={13} />
				add more
			</button>
		</div>
	</header>

	{#if groupId && !serverGroup}
		<div class="surface p-5 text-sm" style="color: var(--ink-muted);">
			<Loader2 size={14} class="animate-spin inline mr-2" />
			Loading shared group <code>{groupId}</code>…
		</div>
	{:else if members.length < 2}
		<div class="surface p-5 text-sm" style="background: var(--rose-dim); border-color: rgba(187, 119, 68, 0.4);">
			You need at least 2 group members.
			{#if groupId}
				Open <a href="/group/{groupId}/join" class="underline" style="color: var(--rust);">the join link →</a>
			{:else}
				<a href="/" class="underline" style="color: var(--rust);">Go add some →</a>
			{/if}
		</div>
	{:else}
		<div class="grid grid-cols-2 md:grid-cols-4 gap-3 anim-fade-up">
			<Stat label="group size" value={members.length} hint="uploaded friends" />
			<Stat label="total ratings" value={totalRated.toLocaleString()} hint="across the group" />
			<Stat label="strategy" value={strategy.replace(/_/g, ' ')} hint="how votes combine" />
			<Stat label="mode" value={mode} hint="re-rank tone per member" />
		</div>

		{#if serverGroup}
			{#if winner}
				{@const winnerClean = winner.title.replace(/\s*\((19|20)\d{2}\)\s*$/, '')}
				<!-- The board out front. Highest net vote gets its name in lights. -->
				<div class="marquee-board px-8 py-5 text-center anim-fade-up" style="animation-delay: 40ms;">
					<div class="board text-[11px]" style="color: var(--ink-faint); letter-spacing: 0.3em;">
						★ Now Showing ★
					</div>
					<div
						class="board neon-amber font-semibold my-1.5"
						style="font-size: clamp(1.4rem, 3.4vw, 2.3rem); letter-spacing: 0.06em;"
					>
						{winnerClean}
					</div>
					<div class="flex justify-center gap-6 mono text-[11px]" style="letter-spacing: 0.16em;">
						<span class="neon-green">+{winner.up} YES</span>
						<span style="color: {winner.veto > 0 ? 'var(--rust)' : 'var(--ink-faint)'};">−{winner.veto} VETO</span>
					</div>
				</div>
			{/if}
			<div class="surface p-4 flex flex-wrap items-center gap-4 text-sm anim-fade-up" style="animation-delay: 40ms;">
				<label class="flex items-center gap-2">
					<span class="text-[10px] uppercase tracking-[0.14em]" style="color: var(--ink-faint);">
						you are
					</span>
					<select bind:value={voterName} class="input" style="padding: 0.3rem 0.5rem; width: auto;">
						{#each serverGroup.members as m (m.name)}
							<option value={m.name}>{m.name}</option>
						{/each}
					</select>
				</label>
				{#if !winner}
					<span class="text-[11px]" style="color: var(--ink-faint);">
						Vote yes / veto on each ticket — the highest net vote goes up on the marquee.
					</span>
				{/if}
			</div>
		{/if}

		<div class="surface p-5 space-y-4 anim-fade-up" style="animation-delay: 60ms;">
			<StrategySelector {strategies} bind:value={strategy} />
			<ModeSelector {modes} bind:value={mode} label="Re-rank mode (applied per member)" />
			<div class="flex flex-wrap items-center gap-4 text-sm pt-1">
				<label class="flex items-center gap-2">
					<span class="text-[10px] uppercase tracking-[0.14em]" style="color: var(--ink-faint);">
						top n
					</span>
					<input
						type="number"
						min="1"
						max="100"
						bind:value={topN}
						class="input"
						style="width: 5rem; padding: 0.3rem 0.5rem;"
					/>
				</label>
				{#if stale}
					<span class="text-[11px] ml-auto" style="color: var(--brand);">
						settings changed — results below are from the previous run
					</span>
				{/if}
				<button
					class="btn btn-secondary {stale ? '' : 'ml-auto'}"
					style={stale ? 'box-shadow: 0 0 0 2px rgba(201, 165, 84, 0.5);' : ''}
					onclick={run}
					disabled={busy}
				>
					{#if busy}
						<Loader2 size={14} class="animate-spin" />
						computing
					{:else}
						<Sparkles size={14} />
						recompute
					{/if}
				</button>
			</div>
		</div>

		{#if error}
			<p class="text-sm rounded-md px-3 py-2" style="background: var(--rose-dim); border: 1px solid rgba(187, 119, 68, 0.4); color: #dda679;">{error}</p>
		{/if}

		{#if watchlistResult && watchlistResult.items.length > 0}
			<section class="surface p-5 space-y-3 anim-fade-up" style="animation-delay: 100ms;">
				<div class="flex items-baseline justify-between gap-3 flex-wrap">
					<div class="flex items-center gap-2">
						<span class="chip chip-violet">
							<UsersIcon size={11} />
							Shared watchlist
						</span>
						<span class="text-xs" style="color: var(--ink-muted);">
							{watchlistResult.items.length} film{watchlistResult.items.length === 1 ? '' : 's'}
							already on {watchlistResult.n_with_watchlist}/{members.length} watchlists
						</span>
					</div>
					<span class="text-[11px]" style="color: var(--ink-faint);">
						the most direct signal: everyone here already <em>wants</em> to see these
					</span>
				</div>

				<div class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2">
					{#each watchlistExpanded ? watchlistResult.items : watchlistResult.items.slice(0, 12) as it (it.movie_id)}
						{@const titleClean = it.title.replace(/\s*\((19|20)\d{2}\)\s*$/, '')}
						{@const yearMatch = it.title.match(/\((19|20)\d{2}\)\s*$/)}
						<div
							class="surface p-3 flex items-center gap-3"
							style="background: rgba(143, 175, 122, 0.07); border-color: rgba(143, 175, 122, 0.25);"
						>
							<div class="flex-1 min-w-0">
								<div class="font-medium text-sm leading-tight truncate" title={it.title}>
									{titleClean}
								</div>
								{#if yearMatch}
									<div class="text-[10px] mono" style="color: var(--ink-faint);">
										{yearMatch[0].replace(/[()]/g, '').trim()}
									</div>
								{/if}
								<div class="mt-1.5 flex items-center gap-1">
									{#each it.members as m (m)}
										<div title={m}>
											<Avatar name={m} size={16} />
										</div>
									{/each}
									<span class="ml-1 text-[10px] mono" style="color: var(--ink-faint);">
										{it.count}/{members.length}
									</span>
								</div>
							</div>
						</div>
					{/each}
				</div>

				{#if watchlistResult.items.length > 12}
					<div class="flex justify-center pt-1">
						<button
							class="btn btn-ghost btn-pill text-xs"
							onclick={() => (watchlistExpanded = !watchlistExpanded)}
						>
							{watchlistExpanded ? 'show fewer' : `show all ${watchlistResult.items.length}`}
						</button>
					</div>
				{/if}
			</section>
		{:else if membersWithWatchlist.length < 2 && topResult}
			<p class="text-xs px-1" style="color: var(--ink-faint);">
				Tip: upload <code>watchlist.csv</code> for each member to surface films everyone already wants to watch.
			</p>
		{/if}

		{#if busy && !topResult}
			<div class="grid grid-cols-1 md:grid-cols-2 gap-3 anim-fade-up">
				{#each Array(8) as _, i (i)}
					<div class="surface p-3 flex gap-3 h-32">
						<Shimmer width="6rem" height="100%" rounded="0.5rem" />
						<div class="flex-1 space-y-2 py-1">
							<Shimmer height="1rem" width="70%" />
							<Shimmer height="0.75rem" width="40%" />
							<Shimmer height="2rem" width="100%" />
						</div>
					</div>
				{/each}
			</div>
		{/if}

		{#if topResult}
			<div class="relative">
				<div class="flex flex-wrap items-center gap-1.5 text-sm lg:pr-56" role="tablist">
					<button
						class="btn btn-pill text-xs board"
						style="letter-spacing: 0.14em; {view === 'top'
							? 'color: var(--brand); border: 1px solid rgba(201, 165, 84, 0.6); background: rgba(201, 165, 84, 0.06); text-shadow: 0 0 8px rgba(201, 165, 84, 0.55); box-shadow: 0 0 10px rgba(201, 165, 84, 0.18);'
							: 'color: var(--ink-faint); border: 1px solid var(--border);'}"
						role="tab"
						aria-selected={view === 'top'}
						onclick={() => (view = 'top')}
					>
						<Sparkles size={12} />
						Top picks
					</button>
					<button
						class="btn btn-pill text-xs board"
						style="letter-spacing: 0.14em; {view === 'argue'
							? 'color: var(--rust); border: 1px solid rgba(187, 119, 68, 0.6); background: rgba(187, 119, 68, 0.06); text-shadow: 0 0 8px rgba(187, 119, 68, 0.55); box-shadow: 0 0 10px rgba(187, 119, 68, 0.16);'
							: 'color: rgba(207, 138, 82, 0.65); border: 1px solid rgba(187, 119, 68, 0.3);'}"
						role="tab"
						aria-selected={view === 'argue'}
						onclick={() => (view = 'argue')}
						disabled={!argueResult || argueResult.recommendations.length === 0}
						title="Films one of you loves and another would skip"
					>
						<UsersIcon size={12} />
						Argue about this
						{#if argueResult}
							<span class="text-[10px] mono ml-1" style="color: var(--ink-faint);">
								{argueResult.recommendations.length}
							</span>
						{/if}
					</button>
					<button
						class="btn btn-pill text-xs board"
						style="letter-spacing: 0.14em; {view === 'seen'
							? 'color: var(--green); border: 1px solid rgba(143, 175, 122, 0.65); background: rgba(143, 175, 122, 0.06); text-shadow: 0 0 8px rgba(95, 135, 95, 0.6); box-shadow: 0 0 10px rgba(95, 135, 95, 0.18);'
							: 'color: rgba(143, 175, 122, 0.6); border: 1px solid rgba(143, 175, 122, 0.28);'}"
						role="tab"
						aria-selected={view === 'seen'}
						onclick={() => (view = 'seen')}
						disabled={!seenResult || seenResult.recommendations.length === 0}
						title="Strict: no member has rated or watched any film in this list"
					>
						<Eye size={12} />
						Nobody's seen
					</button>
				</div>
				<span
					class="chalk absolute right-0 -top-1 hidden lg:block"
					style="font-size: 1.15rem; transform: rotate(-2deg);"
					aria-hidden="true"
				>
					the “argue” tab is where friendships end&nbsp;&nbsp;↴
				</span>
				<p class="text-[11px] mt-1.5" style="color: var(--ink-muted);">
					{#if view === 'argue'}
						sorted by per-member score spread — high stakes movie nights live here
					{:else if view === 'seen'}
						strict — drops any film any member has rated or watched (vs. Top picks, which leaks)
					{:else}
						may include films one of you has seen (per-member exclude)
					{/if}
				</p>
			</div>

			{#if activeResult && activeResult.recommendations.length > 0}
				<div class="grid grid-cols-1 md:grid-cols-2 gap-3">
					{#each activeResult.recommendations as r, i (r.movie_id)}
						<RecommendationCard
							rank={i + 1}
							movieId={r.movie_id}
							title={r.title}
							score={r.score}
							explanation={r.explanation}
							breakdown={r.breakdown}
							per_member_score={r.per_member_score}
							fairness={r.fairness}
							memberOrder={members.map((m) => m.name)}
							votes={serverGroup?.votes?.[r.movie_id]}
							voterName={serverGroup ? voterName : undefined}
							onVote={serverGroup ? handleVote : undefined}
						/>
					{/each}
				</div>

				<!-- Pagination: ask the rerank for a longer list. Same quality
				     ordering — items 13–24 are just the next-best candidates.
				     Capped at 100 by the API. -->
				{#if topN < 100 && activeResult.recommendations.length >= topN}
					<div class="flex justify-center pt-1">
						<button
							class="btn btn-ghost btn-pill text-xs"
							onclick={() => {
								topN = Math.min(topN + 12, 100);
								run();
							}}
							disabled={busy}
						>
							{#if busy}
								<Loader2 size={12} class="animate-spin" />
								loading…
							{:else}
								Show {Math.min(12, 100 - topN)} more
							{/if}
						</button>
					</div>
				{/if}
			{:else if view === 'argue'}
				<div class="surface p-5 text-sm" style="color: var(--ink-muted);">
					Not enough disagreement in this candidate pool to surface contested picks. Try a different mode (e.g. <code>serendipitous</code>) or a larger group.
				</div>
			{:else if view === 'seen'}
				<div class="surface p-5 text-sm" style="color: var(--ink-muted);">
					No strictly-unseen picks survived the filter. Either every candidate slot is occupied
					by something at least one of you has seen, or the candidate pool was too small.
				</div>
			{/if}
		{/if}
	{/if}
</section>
