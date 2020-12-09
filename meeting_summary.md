
*next meeting: CT 12/8周二晚上8点*

# Standardization

agent和pde里面有些共用的参数和函数，我们可以统一一下，方便调用，沟通和结果比较

## Parameters

## ✅ b and k

 > Use your results from the midterm checkpoint to choose b and k near the phase transition boundary.

 use b=7, k=0.1

## ✅ N, T, I0

平常操作用N=1000, I=1
出图可以用N=10000, I=10
T = 100

## Functions

### Q1

> If infected individuals start at the center of the square, how does the total number of individuals infected throughout the simulation change with the parameter p?

- 为了回答这个问题我们可以画个图，横轴是t，纵轴还是infectious proportion i(t), 然后用不同的p画i(t)，也就是把p作为legend，画多条颜色不同的线。
  - 管这个图叫**step effect plot**好了
  - 对应的画图函数定义成`plot_step_effect(i_ts, ps)`,  `i_ts.shape = (np, T)`, `i_ts`每行`i_t`是某个p时的i(t), `np = len(ps)`是试验的p的个数, `ps`用来画legend。agent和pde都可以用这个画图函数，只要最后result统一成上面的形式。
  - 参考 agent_seir_simulation.py 里的plot.lines。colormap选OrRd，跟midterm的i(t)的橙色比较接近


### Q2

> Choose an interesting parameter of p using question 1. How does the simulation qualitatively differ when the initial infected individuals start in a single corner of the square vs. the center of the square vs. being randomly spread out?

- ✅ 怎么定义interesting?
  - ❌使得i(t)的峰值最大的p   p = argmax_p max_t i_p(t)  
    - 此时p太大，跟起始位置关系就不大了
  - p=0, p=.5, =1

- 选好the interesting p后，我们也可以画i(t)来回答这个问题。legend换成上面三个, i.e. corner, center, randomly spread out.
  - randomly spread out的话，我觉得可以random sample多一点。比如，如果三种legend对应颜色红绿蓝，则最后的图是n1条红线，n2条绿线，n3条蓝线，n是random sample的次数。
  - 管这个line plot叫**location effect plot**好了
  - 对应的画图函数可以定义成`plot_loc_effect(i_ts)`，where `i_ts.shape = (n1+n2+n3, T)`, 分别是corner, center，random spread out的试验次数。同理，agend和pde都可以用这个画图函数


- 静态二维图
  - 三个case，每个case选4个有意义的t，画成3*4的subplots
  - 管他叫plot_loc_effect_2d(i_ts)


- Optional：动图，每帧对应时刻t
  - agent的每帧是N个散点在[0,1] * [0,1]的scatter plot，s,i,r分别用blue, red, green吧, red有危险的意味，green表示健康，不会被感染
    - 可以参考knn那节的画图函数，用来做每帧。
  - pde的每帧是M*M grid, 每格子颜色是从蓝到红的scale, i(x,y, t)值越大，颜色越红，反之越蓝。
    - agend_based_models.ipynb这节的画图函数应该可以直接拿来用，数据结构也差不多。不过那里的颜色只是binary。我们想用color scale，还要改改。
  - plot_loc_effect_2d_gif

# Models

## Agent-based model

✅和pde共有的问题：怎么选b, k, N, T, I0

✅单独的待解决问题：怎么选interesting p

## PDE

> Note that you can turn this into a system of ODEs by vectorizing the 2-dimensional arrays s, i and r.

✅目前我们讨论结果是还是用solve_ivp()来做，
✅就是f(t,y)有点难写。有想法或困难及时交流！

## Midterm Variations

### Varying parameters
✅agent
ode里不好对每个t设置设置不同的b，k。还是用piece-wise constant来做比较方便。agent也用此方法。

### SEIR
✅好做，但由于加了参数a和e0，需要画的图有点多。

### Fitting
目前只有daily cases的数据来源比较靠谱。用来estimate `s`.
可能要分piece-wise去fit。

# 分工

迥仪
  - ✅可以先确定b, k, N, T, I0，补全上面文档对应section
  - continue on agent
    - 可以先写`plot_step_effect`和`plot_loc_effect`

学弟
  - continue on pde

dennis
  - ✅写agent2.py，用一个attribute state表示SIR状态
    - 实际上写了agent_pop.py, 新建了一个class Population() , 避免了agent()的loop和内存占用。速度应该会快一点

  - 写三个variations的simulation
    - ✅agent_varparm
      - N=10000的图更granular，但耗时 ✅已解决这个问题。见agent_pop.py里的infect()
        - 用N=1000做草图
        - 确定形态后，用N=10000出图
      - findings
        - social distancing can flatten the curve
          - postpone peak arrival
          - lowers peak value, but not significant
          - peak value also depends on k
        - drug effect is significant to reduce i
        - mutation increases i,
          - mutation in early stage is a disaster, high peak may hit hospital's capacity
          - in later stage is ok, since r is high, s is low
      - Optional:
        - 保留最开始的sir结果，从effect生效时间开始simlate后面的simulation
          - sirs2 = sirs, sirs2[t:] = pop.simulation()
        - 做 pre-peak, post-peak
        - 给图中加竖直线标注ax.axvline(group_mean, ls='--', color='r')
        - 增加T

    - ✅ode_varparm
        - ❓p怎么加进ode
    - ✅ agent_seir
      - 选b=1, k=0.01, 画plot_f_effect,
        - 每条线是i(t), 对应一个f，    color scale
      - 2-D phase diagram
        - pairwise
      - optional:
        - 3-D phase diagram in html
        - ❓save的图不一样

    - ✅ ode_seir
      - effect of f
        - large f is equivalent to sir
        - small f flattens i(t), many stuck in state e

    - fitting
      -

  - 改module和更新push.yml 
