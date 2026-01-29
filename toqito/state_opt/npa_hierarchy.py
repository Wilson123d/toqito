"""Generates the NPA constraints."""

from collections import namedtuple
from itertools import product

import cvxpy

Symbol = namedtuple("Symbol", ["player", "question", "answer"], defaults=["", None, None])
IDENTITY_SYMBOL = Symbol("", None, None)  # Explicit identity symbol
PLAYERS = ("Alice", "Bob")


def _reduce(word: tuple[Symbol, ...]) -> tuple[Symbol, ...]:
    """Reduce an operator word to its canonical form using NPA rules.

    Identity: I*S = S*I = S, I*I = I
    Commutation: Alice operators commute with Bob operators. Canonical form: A...AB...B
    Orthogonality: P_x,a P_x,b = 0 if a != b (for same player x)
    Idempotence: P_x,a P_x,a = P_x,a (for same player x)
    """
    if not word:
        return ()

    # Initial pass to filter out identities IF other ops are present
    current_list = [s for s in word if s != IDENTITY_SYMBOL]
    if not current_list:  # Original word was all identities or empty
        return (IDENTITY_SYMBOL,) if any(s == IDENTITY_SYMBOL for s in word) else ()

    # Canonical player order (Alice then Bob), preserving original relative internal order
    alice_ops = [s for s in current_list if s.player == "Alice"]
    bob_ops = [s for s in current_list if s.player == "Bob"]
    current_list = alice_ops + bob_ops  # This is now a list of Symbol objects

    # Iteratively apply reduction rules until no more changes occur
    while True:
        len_before_pass = len(current_list)
        next_pass_list = []
        idx = 0
        made_change_in_pass = False

        while idx < len(current_list):
            s_x = current_list[idx]

            if idx + 1 < len(current_list):
                s_y = current_list[idx + 1]
                # Only apply if s_x and s_y are from the same player.
                if s_x == s_y and s_x.player in PLAYERS:  # s_x != IDENTITY_SYMBOL
                    next_pass_list.append(s_x)
                    idx += 2  # Consumed s_x, s_y; added s_x
                    made_change_in_pass = True
                    continue
                # Rule 2: Orthogonality (S_x,a S_x,b = 0 if a!=b, for same player and question)
                elif (
                    s_x.player == s_y.player
                    and s_x.player in PLAYERS  # Ensure not identity
                    and s_x.question == s_y.question
                    and s_x.answer != s_y.answer
                ):
                    return ()  # Entire word becomes zero
                else:
                    # No reduction for this pair, keep s_x
                    next_pass_list.append(s_x)
                    idx += 1
            else:
                # Last element, just append it
                next_pass_list.append(s_x)
                idx += 1

        current_list = next_pass_list
        if not made_change_in_pass and len(current_list) == len_before_pass:  # Stable
            break

    return tuple(current_list) if current_list else ()


def _parse(k_str: str) -> tuple[int, set[tuple[int, int]]]:
    if not k_str:  # Explicitly handle empty string input for k_str
        raise ValueError("Input string k_str cannot be empty.")
    parts = k_str.split("+")
    if not parts[0] or parts[0] == "":  # Check if the first part (base_k) is empty
        raise ValueError("Base level k must be specified, e.g., '1+ab'")
    try:
        base_k = int(parts[0])
    except ValueError as e:
        raise ValueError(f"Base level k '{parts[0]}' is not a valid integer: {e}") from e

    conf = set()
    if len(parts) == 1 and base_k >= 0:  # e.g. "0", "1"
        pass  # conf remains empty, which is correct.

    for val_content in parts[1:]:  # Process each part after the base_k
        cnt_a, cnt_b = 0, 0
        if not val_content:  # Handles "1++ab" -> parts like '', skip these
            continue
        # If val_content is an empty string (e.g., from "0+", "1++a"),
        # cnt_a and cnt_b will remain 0, and (0,0) will be added to conf.
        for char_val in val_content:  # Loop over empty string does nothing
            if char_val == "a":
                cnt_a += 1
            elif char_val == "b":
                cnt_b += 1
            else:
                raise ValueError(
                    f"Invalid character '{char_val}' in k string component "
                    + f"'{val_content}'. Only 'a' or 'b' allowed after base k."
                )
        conf.add((cnt_a, cnt_b))
    return base_k, conf


def _gen_words(k: int | str, a_out: int, a_in: int, b_out: int, b_in: int) -> list[tuple[Symbol, ...]]:
    # Symbols for non-identity measurements (last outcome is dependent)
    alice_symbols = [Symbol("Alice", x, a) for x in range(a_in) for a in range(a_out - 1)]
    bob_symbols = [Symbol("Bob", y, b) for y in range(b_in) for b in range(b_out - 1)]

    words = set([(IDENTITY_SYMBOL,)])  # Start with identity operator

    k_int = k
    configurations = set()

    if isinstance(k, str):
        k_int, configurations = _parse(k)

    # Loop 1: Generate words up to length k_int from the hierarchy
    for length in range(0, k_int + 1):  # Lengths 1, ..., k_int
        for alice_len in range(length + 1):
            bob_len = length - alice_len

            # Generate Alice's part
            # If alice_len is 0, product yields one item: ()
            for word_a_tuple in product(alice_symbols, repeat=alice_len):
                reduced_a = _reduce(word_a_tuple)
                # Alice's part (non-empty originally) reduced to zero
                if reduced_a == () and alice_len > 0:
                    continue

                # Generate Bob's part
                # If bob_len is 0, product yields one item: ()
                for word_b_tuple in product(bob_symbols, repeat=bob_len):
                    reduced_b = _reduce(word_b_tuple)
                    # Bob's part (non-empty originally) reduced to zero
                    if reduced_b == () and bob_len > 0:
                        continue

                    if not reduced_a and not reduced_b:  # Both parts are empty (e.g. alice_len=0, bob_len=0)
                        # This means the total length of operators is 0.
                        final_word = (IDENTITY_SYMBOL,)
                    else:
                        # _reduce will put Alice operators before Bob operators if somehow mixed,
                        # and apply rules. It also handles identity filtering if I was part of word.
                        # Here, reduced_a + reduced_b is already A...AB...B (or just A...A or B...B).
                        final_word = _reduce(reduced_a + reduced_b)
                    words.add(final_word)

    # Loop 2: Add words from specific configurations (e.g., "1+ab" means k_int=1, configurations={(1,1)})
    for alice_len_conf, bob_len_conf in configurations:
        if alice_len_conf == 0 and bob_len_conf == 0 and k_int == 0 and (IDENTITY_SYMBOL,) in words:
            pass  # The set `words` will handle duplicates from k_int loop vs config loop.

        for word_a_tuple in product(alice_symbols, repeat=alice_len_conf):
            reduced_a = _reduce(word_a_tuple)
            if reduced_a == () and alice_len_conf > 0:
                continue

            for word_b_tuple in product(bob_symbols, repeat=bob_len_conf):
                reduced_b = _reduce(word_b_tuple)
                if reduced_b == () and bob_len_conf > 0:
                    continue

                # Combine and add as in the main loop
                # Both parts are empty (e.g. alice_len_conf=0, bob_len_conf=0)
                if not reduced_a and not reduced_b:
                    # Should not happen if _parse filters (0,0) from conf
                    final_word = (IDENTITY_SYMBOL,)

                else:
                    final_word = _reduce(reduced_a + reduced_b)

                words.add(final_word)

    # If `words` contains `()`, filter it out before converting to list.
    words = {w for w in words if w != ()}
    # Convert set to list, then sort.
    # Make sure (IDENTITY_SYMBOL,) is always at index 0.
    list_of_words = list(words)
    list_of_words.remove((IDENTITY_SYMBOL,))
    # Sort remaining words: typically by length, then by content.
    # Sorting tuples of Symbols needs a consistent key.
    # repr(s) can give a consistent string for sorting.
    list_of_words.sort(key=lambda w: (len(w), tuple(repr(s) for s in w)))
    return [(IDENTITY_SYMBOL,)] + list_of_words


def _is_zero(word: tuple[Symbol, ...]) -> bool:
    # An empty tuple after reduction means the operator product is zero.
    return len(word) == 0


def _is_identity(word: tuple[Symbol, ...]) -> bool:
    return word == (IDENTITY_SYMBOL,)


def _is_meas(word: tuple[Symbol, ...]) -> bool:
    # Expects a reduced word: (Alice_Symbol, Bob_Symbol)
    if len(word) == 2:
        s_a, s_b = word
        return s_a.player == "Alice" and s_b.player == "Bob"
    return False


def _is_meas_on_one_player(word: tuple[Symbol, ...]) -> bool:
    # Expects a reduced word: (Alice_Symbol,) or (Bob_Symbol,)
    if len(word) == 1:
        s = word[0]
        return s.player in PLAYERS  # Excludes IDENTITY_SYMBOL
    return False


# _get_nonlocal_game_params remains the same as in npa_constraints_fix
def _get_params(
    assemblage: dict[tuple[int, int], cvxpy.Variable], referee_dim: int = 1
) -> tuple[int, int, int, int]:
    a_in, b_in = max(assemblage.keys())
    a_in += 1
    b_in += 1
    operator = next(iter(assemblage.values()))
    a_out = operator.shape[0] // referee_dim
    b_out = operator.shape[1] // referee_dim
    return a_out, a_in, b_out, b_in


def npa_constraints(assemblage, k=1, referee_dim=1, no_signaling: bool = True):
    a_out, a_in, b_out, b_in = _get_params(assemblage, referee_dim)
    words = _gen_words(k, a_out, a_in, b_out, b_in)
    if not words:
        raise ValueError("Generated word list is empty.")
    dim, dR = len(words), referee_dim
    word_to_idx = {word: i for i, word in enumerate(words)}
    
    moment_matrix = cvxpy.Variable((dR * dim, dR * dim), hermitian=True)
    constraints = [moment_matrix >> 0]
    
    rho_R = moment_matrix[0:dR, 0:dR]
    constraints.append(cvxpy.trace(rho_R) == 1)

    # Link moment matrix to the assemblage for basis operators
    seen_products = {}
    for i, word_i in enumerate(words):
        for j, word_j in enumerate(words):
            if i > j: continue
            block = moment_matrix[i*dR:(i+1)*dR, j*dR:(j+1)*dR]
            prod = _reduce(tuple(reversed(word_i)) + word_j)
            
            if not prod: constraints.append(block == 0); continue
            if prod in seen_products:
                p_i, p_j = seen_products[prod]
                constraints.append(block == moment_matrix[p_i*dR:(p_i+1)*dR, p_j*dR:(p_j+1)*dR]); continue
            
            seen_products[prod] = (i, j)
            if prod == (IDENTITY_SYMBOL,):
                constraints.append(block == rho_R)
            elif len(prod) == 2 and prod[0].player=="Alice" and prod[1].player=="Bob":
                s_a, s_b = prod; x, a, y, b = s_a.question, s_a.answer, s_b.question, s_b.answer
                constraints.append(block == assemblage[x, y][a*dR:(a+1)*dR, b*dR:(b+1)*dR])
            elif len(prod) == 1 and prod[0].player in PLAYERS:
                s = prod[0]; x, a = s.question, s.answer
                if s.player == "Alice":
                    constraints.append(block == sum(assemblage[x,0][a*dR:(a+1)*dR, b_idx*dR:(b_idx+1)*dR] for b_idx in range(b_out)))
                else:
                    constraints.append(block == sum(assemblage[0,x][a_idx*dR:(a_idx+1)*dR, a*dR:(a+1)*dR] for a_idx in range(a_out)))

    # CRITICAL FIX: Add constraints for the dependent outcomes
    for x, y in product(range(a_in), range(b_in)):
        # Dependent Alice outcome
        a_last = a_out - 1
        sum_K_a = sum(assemblage[x,y][a*dR:(a+1)*dR, b*dR:(b+1)*dR] for a in range(a_out-1) for b in range(b_out))
        K_a_last = sum(assemblage[x,y][a_last*dR:(a_last+1)*dR, b*dR:(b+1)*dR] for b in range(b_out))
        constraints.append(K_a_last == rho_R - sum_K_a)
        
        # Dependent Bob outcome
        b_last = b_out - 1
        sum_K_b = sum(assemblage[x,y][a*dR:(a+1)*dR, b*dR:(b+1)*dR] for b in range(b_out-1) for a in range(a_out))
        K_b_last = sum(assemblage[x,y][a*dR:(a+1)*dR, b_last*dR:(b_last+1)*dR] for a in range(a_out))
        constraints.append(K_b_last == rho_R - sum_K_b)

    # Final assemblage positivity and normalization constraints
    for x, y in product(range(a_in), range(b_in)):
        for a, b in product(range(a_out), range(b_out)):
            constraints.append(assemblage[x, y][a*dR:(a+1)*dR, b*dR:(b+1)*dR] >> 0)
        constraints.append(sum(assemblage[x,y][a*dR:(a+1)*dR, b*dR:(b+1)*dR] for a,b in product(range(a_out),range(b_out))) == rho_R)
    if no_signaling:
        # No-signaling constraints on assemblage - ALWAYS APPLY
        # Bob's marginal rho_B(b|y) = Sum_a K_xy(a,b) must be independent of x
        for y_bob_in in range(b_in):
            for b_bob_out in range(b_out):
                sum_over_a_for_x0 = sum(
                    assemblage[0, y_bob_in][
                        a * referee_dim : (a + 1) * referee_dim, b_bob_out * referee_dim : (b_bob_out + 1) * referee_dim
                    ]
                    for a in range(a_out)
                )
                for x_alice_in in range(1, a_in):
                    sum_over_a_for_x_current = sum(
                        assemblage[x_alice_in, y_bob_in][
                            a * referee_dim : (a + 1) * referee_dim,
                            b_bob_out * referee_dim : (b_bob_out + 1) * referee_dim,
                        ]
                        for a in range(a_out)
                    )
                    constraints.append(sum_over_a_for_x0 == sum_over_a_for_x_current)

        # Alice's marginal rho_A(a|x) = Sum_b K_xy(a,b) must be independent of y
        for x_alice_in in range(a_in):
            for a_alice_out in range(a_out):  # For each Alice outcome a
                sum_over_b_for_y0 = sum(
                    assemblage[x_alice_in, 0][
                        a_alice_out * referee_dim : (a_alice_out + 1) * referee_dim,
                        b * referee_dim : (b + 1) * referee_dim,
                    ]
                    for b in range(b_out)
                )
                for y_bob_in in range(1, b_in):
                    sum_over_b_for_y_current = sum(
                        assemblage[x_alice_in, y_bob_in][
                            a_alice_out * referee_dim : (a_alice_out + 1) * referee_dim,
                            b * referee_dim : (b + 1) * referee_dim,
                        ]
                        for b in range(b_out)
                    )
                    constraints.append(sum_over_b_for_y0 == sum_over_b_for_y_current)    

    return constraints
