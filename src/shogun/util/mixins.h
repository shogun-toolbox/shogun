#ifndef MIXINS_H
#define MIXINS_H

#include <shogun/lib/sg_types.h>
#include <utility>

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace shogun
{
	template <
	    class D, template <typename> class U, template <typename> class... Ts>
	struct mutator;

	template <class D, template <typename> class... Ts>
	class composition : public Ts<mutator<D, Ts, Ts...>>...
	{
	public:
		composition() : Ts<mutator<D, Ts, Ts...>>()...
		{
		}

		composition(const composition<D, Ts...>& orig)
		    : Ts<mutator<D, Ts, Ts...>>(orig)...
		{
		}

		composition(composition<D, Ts...>&& orig)
		    : Ts<mutator<D, Ts, Ts...>>(std::move(orig))...
		{
		}

		virtual ~composition()
		{
		}

	protected:
		template <template <typename> class I>
		using mixin_t = I<mutator<D, I, Ts...>>;
	};

	template <
	    class D, template <typename> class U, template <typename> class... Ts>
	struct mutator
	{
	private:
		using composition_t = composition<D, Ts...>;
		using mixin_list_t = TemplateTypes<Ts...>;
		template <template <typename> class I>
		using has_type = typename mixin_list_t::template has<I>;

		// generate mutator type for mixin
		template <template <typename> class I>
		using mutator_t = mutator<D, I, Ts...>;
		// generate full mixin type
		template <template <typename> class I>
		using mixin_t = I<mutator_t<I>>;

		template <template <typename> class I>
		struct friend_impl
		{
			static_assert(
			    mixin_t<U>::friends::template has<I>::value,
			    "Mixin is not a friend!");

			using type =
			    std::conditional_t<has_type<I>::value, mixin_t<I>, None>;
		};

		template <template <typename> class I>
		struct requirement_impl
		{
			static_assert(
			    mixin_t<U>::requirements::template has<I>::value,
			    "Mixin is not a requirement!");

			static_assert(
			    mixin_list_t::template has<I>::value,
			    "Composition mixins are not compatible!");

			using type = mixin_t<I>;
		};

	public:
		using this_t = mixin_t<U>;

		using derived_t =
		    std::conditional_t<std::is_same<D, None>::value, composition_t, D>;

		template <template <typename> class I>
		using friend_t = typename friend_impl<I>::type;

		template <template <typename> class I>
		using requirement_t = typename requirement_impl<I>::type;

		template <template <typename> class I>
		requirement_t<I>& mutate()
		{
			auto derived = static_cast<composition_t*>(this);
			return static_cast<mixin_t<I>&>(*derived);
		}

		virtual ~mutator()
		{
		}
	};

	enum MixinOptionType
	{
		Requirements,
		Friends
	};

	template <template <typename> class... Ts>
	struct requires : TemplateTypes<Ts...>
	{
		static constexpr MixinOptionType option_type = Requirements;
	};

	template <template <typename> class... Ts>
	struct friends_with : TemplateTypes<Ts...>
	{
		static constexpr MixinOptionType option_type = Friends;
	};

	template <MixinOptionType ot, typename... Ts>
	struct find_option
	{
		using type = TemplateTypes<>;
	};

	template <MixinOptionType ot, typename T, typename... Ts>
	struct find_option<ot, T, Ts...>
	    : std::conditional<
	          T::option_type == ot, T, typename find_option<ot, Ts...>::type>
	{
	};

	template <typename M, typename... Options>
	struct mixin : public M
	{
		friend M;
		virtual ~mixin(){};

	private:
		using requirements =
		    typename find_option<Requirements, Options...>::type;
		using friends = typename find_option<Friends, Options...>::type;
	};
} // namespace shogun
#endif // DOXYGEN_SHOULD_SKIP_THIS
#endif // MIXINS_H
