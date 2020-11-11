

# 统一格式和参数

## 画图

画图就两张

* 学弟画好的那张S,I,R在不同b,k时的plot, 我们管它叫facet plot

* 另外一张是 t=T时的phase diagram。

* （有空的话我们可以再加个phase diagram在时间T上的GIF)。

图片记得加labels, title, legend

保存到`/project-group-4/output/modelname_plotname.png`，modelname = agent/ode, plotname = facet_plot / phase_diagram


## simulation
simulation里T，bs，ks的范围，我们选到such that final infectious fraction i(T) 能比较均匀跨越(0,1)，对应的也就是在t=T的phase diagram 里，颜色比较均匀地从浅到深渐变，这样在report里比较好下结论（哪些b,k使得t=T时所有/部分人感染，critical points, etc）

# 这阶段的分工：

## 学弟

1. 为了后半学期代码reproducibility，学弟可以优化一下这部分代码：将每次`for b … for …k` 的s,i,r结果存成一个ndarray (参考agent_simulation.py), 然后用这个ndarray来做后续的分析，比如画图。简单修改的话，就在plot_sim()里面加一行类似`results = results.append(np.array([s,i,r]))`的代码。
1. 画phase diagram，可以参考`ageng_simulation.py`的相应代码，用results画的话很方便
1. 保存图片到`/output`
1. 将`ode.ipynb`转成`ode_simulation.py`
1. 根据simulation的结果，可以开始完成`report.ipynb` 里section 3 ODE的部分，文字+插图
1. 想一个variation，写在`report.ipynb` section 4

## 迥仪

1. 目前我们还不清楚b到底是啥，在这之前迥仪可以先看看`agent_simulation.py`里我写的`count_sir()`，以及后面`for b … for k…`里面`results=…`的逻辑。明早确定b的用法后，我们再看看`run_sim()`里面要怎么改。

1. 写用results来画facet plot的代码，代码可以参考学弟的代码。大致伪代码如下
```
f, ax = plt.subplots()
for i,j:
        use results[i,j,:,:] to generate a plot of s i r
        this plot corresponds to parameters bs[i] and ks[j]
```
1. 给`agent_simulation.py`的图加label legend 以及保存到output的代码
1. 确定b后，run一下`agent_simulation.py`就有图了，然后可以开始完成`report.ipynb` 里section 3 Agent的部分，文字+插图
1. 想一个variation，写在`report.ipynb` section 4


## 少雄

1. 问Brad `b`到底怎么用
1. 根据情况修改`agent.py`和`agent_simulation.py`
1. 写完`test.py`
1. 修修补补debug
1. 想一个variation，写在`report.ipynb` section 4


### ps

* 做这种project的simulation -> analysis步骤时，最好把一次考虑了多种参数和需求的simulation的结果存成一个大的ndarray(or导出存起来)，后面每步analysis或者画图，取相应部分来用即可，这样可以避免每次analysis都回去耗时跑simulation

* report用ipynb写的话
  * 相比md的好处是，打公式更方便
  * 相比word的好处是，ipynb图片自带路径。若更新code则自动更新report图片，不用像word那样需要重新插入
