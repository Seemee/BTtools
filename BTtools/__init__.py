Z,Y,X=0,1,2
#from IPython.core.display import display, HTML
#import exactcover
import gzip
import math
from pathlib import Path
from lxml import objectify
from lxml import etree
import numpy as np 
import copy
import re
import glob
import os
import sys
IN_COLAB = 'google.colab' in sys.modules
files=None
if IN_COLAB:
    from google.colab import files
class BTtools:
    def __init__(self, filename=None):
        global files
        print('Burrtools Tools v6.57')
        if filename==None:
            puzzle=etree.Element('puzzle')
            puzzle.set('version','2')
            gridType=etree.SubElement(puzzle, 'gridType')
            gridType.set('type','0')
            etree.SubElement(puzzle, 'colors')
            etree.SubElement(puzzle, 'shapes')
            etree.SubElement(puzzle, 'problems')
            etree.SubElement(puzzle, 'comment')
            # make every node accessible
            xml=etree.tostring(puzzle) 
            self.obj = objectify.fromstring(xml)
            return
        xml=''
        if IN_COLAB:            
            upload = files.upload()
            self.filename,fileBuffer=next(iter(upload.items()))
            xml=gzip.decompress(fileBuffer)
        else:
            self.filename = filename
            if os.path.isdir(filename): #if direcorty is give find latest file
                search_dir = filename
                # remove anything from the list that is not a file (directories, symlinks)
                # thanks to J.F. Sebastion for pointing out that the requirement was a list 
                # of files (presumably not including directories)  
                #files = list(filter(os.path.isfile, glob.glob(search_dir + "*")))
                dirs=glob.glob(search_dir+'*' )
                dirs.sort(key=lambda x: os.path.getmtime(x))
                valid=False
                files=None
                for cdir in dirs[::-1]:
                    files=glob.glob(cdir+'\\*.xmpuzzle' )
                    files.sort(key=lambda x: os.path.getmtime(x))
                    if files!=[]:
                        valid=True
                        break
                if not valid or files==[]:
                    print("Error: No .xmpuzzle files found:",self.filename)
                    raise KeyboardInterrupt
                self.filename=files[-1]
                print('Filename:',self.filename)
            try:
                with gzip.open(self.filename, 'rb') as f:
                    xml=f.read()
                f.close()
            except (IOError, OSError):
                print("Error: Openening file:",self.filename)
                raise KeyboardInterrupt
        self.obj = objectify.fromstring(xml)
    def dump(self,obj=None):
        if obj==None:
            obj=self.obj
        etree.dump(obj)
    @staticmethod
    def separation(obj,moves):
        if hasattr(obj,'separation'):
            ob=obj.separation
            cnt=len(ob.state)-1
            moves.append(cnt)
            #print(cnt,end='.')
            BTtools.separation(ob,moves)
        elif hasattr(obj,'separationInfo'):
            lst=str(obj.separationInfo).split(' ')
            for item in lst:
                if item!='0':
                    moves.append(int(item)-1)
        return moves
    def process(self,problemSelect,ignoreShapes=[],verbose=False):
        voxelCount={}
        for no,voxel in enumerate(self.obj.shapes.voxel):
            coords=self.getAllCoordinates(self.getArray(voxel))            
            voxelCount[no+1]=len(coords[0])
        problem=self.obj.problems.problem[problemSelect-1]
        state=problem.get('state')
        if state==None:
            print("Error: Problem %d is in initial state. No solutions exist!" % problemSelect)
            raise KeyboardInterrupt
        state=int(state)
        if verbose:
            if state<2:
                print("\nProblem solving for P%d hasn't finished yet, the results will be inaccurate!" % problemSelect)
            print('Time spent: %.1f hrs.' % (int(problem.get('time'))/3600) )    
        shapeIndices=[]
        shapeDict={}
        if hasattr(problem.shapes,'shape'):
            for shape in problem.shapes.shape:
                maxCount=shape.get('count')
                if maxCount==None:
                    minCount=shape.get('min')
                    maxCount=shape.get('max')
                else:
                    minCount=shape.get('count')
                group=shape.get('group')
                if group==None:
                    if hasattr(shape,'group'):
                        group=shape.group.get('group')
                    else:
                        group=0
                shapeId=int(shape.get('id'))+1
                voxel=self.obj.shapes.voxel[shapeId-1]
                name=voxel.get('name')
                
                arr=self.getArray(voxel)
                coords=self.getAllCoordinates(arr)
                length=len(coords[0])
                planar=False
                if coords[Z].tolist().count(coords[Z][0]) == length or coords[Y].tolist().count(coords[Y][0]) == length or coords[X].tolist().count(coords[X][0]) == length:
                    planar=True                
                shapeDict[shapeId]={'name':name,'group':group,'frequency':0,'maxUse':0,'voxels':voxelCount[shapeId],'min':minCount,'max':maxCount, 'planar':planar}
                for _ in range(int(maxCount)):
                    shapeIndices.append(shapeId)
        solutions={}
        if not hasattr(problem,'solutions'):
            if verbose:
                print("Warning: No solutions exist!")
            return None,shapeDict
        if not hasattr(problem.solutions,'solution'):
            if verbose:
                print("Warning: No solution for problem P%d exist!" % problemSelect)
            return None,ShapeDict
        solCount=len(problem.solutions.solution)
        print('Processing %d solutions in P%d:'%(solCount,problemSelect))
        lastPercentage=-1
        sortedShapeIndices=sorted(shapeIndices)
        for cnt,solution in enumerate(problem.solutions.solution):
            shapesMultipleUseList=[0]*(sortedShapeIndices[-1]+1)
            moves=self.separation(solution,[])
            totMoves=sum(moves)
            asmNum=solution.get('asmNum')
            shapeList=self.getShapes(solution)
            shapeNumbers=self.getShapeNumbers(solution,shapeIndices)
            solList=[]
            for shape in shapeNumbers:
                if shape not in ignoreShapes:
                    solList.append(shape)
                    shapesMultipleUseList[shape]+=1
                    shapeDict[shape]['frequency']+=1
                if shapesMultipleUseList[shape]>shapeDict[shape]['maxUse']:
                    shapeDict[shape]['maxUse']+=1
            sol=tuple(solList)
            percentage=int((cnt+1)/solCount*1000)/10
            if lastPercentage!=percentage:
                print('Sol no.: %d, %.1f%%'%(cnt+1,percentage),end='\r')
                lastPercentage=percentage
            if asmNum is not None:
                asmNum=int(asmNum)+1
            if sol in solutions:
                solutions[sol]['sols']+=1
                if moves!=[] and solutions[sol]['totMoves']<totMoves:
                    solutions[sol]['totMoves']=totMoves
                    solutions[sol]['moves']=moves
                    solutions[sol]['asmNum']=asmNum
            else:
                solutions[sol]={'sols':1, 'asmNum':asmNum,'totMoves':totMoves,'moves':moves}
        print()
        return solutions,shapeDict
    
    @staticmethod
    def bestSolution(solutions):
        minSols=math.inf
        maxSols=0
        minMoveRange=math.inf
        maxMoveRange=0
        for k,v in solutions.items():
            if v['sols']<minSols:
                minSols=v['sols']
                minAsmNum=v['asmNum']
                moves=v['moves']
                minAssembly=k
                totMoves=v['totMoves']
            elif v['sols']==minSols: # and v['totMoves']<maxMoves:
                minAsmNum=v['asmNum']
                moves=v['moves']
                minAssembly=k
                totMoves=v['totMoves']
            if v['sols']>maxSols:
                maxSols=v['sols']
            if v['totMoves']<minMoveRange:
                minMoveRange=v['totMoves']
            if v['totMoves']>maxMoveRange:
                maxMoveRange=v['totMoves']
                
        bs= {'minSols':minSols,'maxSols':maxSols,'totMoves':totMoves,'moves':moves,'minAssembly':minAssembly,'minAsmNum':minAsmNum,'minMoveRange':minMoveRange,'maxMoveRange':maxMoveRange}
        ret = {k: v for k, v in bs.items()}
        return ret

    def copyProblem(self,problemSelect):
        problem = copy.deepcopy(self.obj.problems.problem[problemSelect-1])
        self.obj.problems.append(problem)
        return len(self.obj.problems.problem)

    def createParts(self,problemSelect):
        voxel=self.obj.shapes.voxel[problemSelect-1]
        identities=self.getIdentities(voxel)
        print('Found %d identities: %s'%(len(identities),', '.join(identities)))
        for no,id in enumerate(identities):
            tmpString=copy.deepcopy(voxel.text)
            for i in identities:
                if i!=id:
                    tmpString=tmpString.replace(i,'_')
            newPart=etree.Element("voxel")
            newPart.text=tmpString
            for k,v in voxel.attrib.items():
                newPart.set(k,v)
            newPart.set('name','part '+str(no+1))
            self.obj.shapes.append(newPart)
        
    @staticmethod
    def createNewProblem(shapeIds,shapeDict,problem,name=''):
        newProblem=etree.Element("problem")
        newProblem.set('name',name)
        shapes=etree.SubElement(newProblem, 'shapes')
        quantities={}
        for shapeId in shapeIds:
            if shapeId not in quantities:
                quantities[shapeId]=0
            quantities[shapeId]+=1
        for shapeId,quantity in quantities.items():
            shape=etree.SubElement(shapes, 'shape')
            shape.set('id',str(shapeId-1))
            shape.set('min',str(quantity))
            shape.set('max',str(quantity))
            shape.set('group',str(shapeDict[shapeId]['group']))
            shapes.append(shape)
        newProblem.append(shapes)
        result=copy.deepcopy(problem.result)
        newProblem.append(result)
        bitmap=copy.deepcopy(problem.bitmap)
        newProblem.append(bitmap)
        return newProblem

    def isolate(self,problemSelect):
        problem=self.obj.problems.problem[problemSelect-1]
        shapeIndices=[]
        for shape in problem.shapes.shape:
            shapeIndices.append(int(shape.get('id'))+1)
        newshapeIndices=[id for id in range(1,len(shapeIndices)+2)]
        resultId=int(problem.result.get('id'))
        print('Original shapes: S%s'%(', S'.join(map(str,[resultId]+shapeIndices))))
        print('New shape id\'s: S%s'%(', S'.join(map(str,newshapeIndices))))
        voxel=[self.obj.shapes.voxel[resultId]]
        voxel[0].set('name','(P%d target shape)'%problemSelect)
        for i,id in enumerate(shapeIndices):
            problem.shapes.shape[i].set('id',str(i+1))
            voxel.append(self.obj.shapes.voxel[id-1])
        problem.result.set('id','0')
        self.obj.problems.problem=[problem]
        self.obj.shapes.voxel=voxel

    @staticmethod
    def getShapeIndices(problem):
        shapeIndices=[]
        for shape in problem.shapes.shape:
            maxCount=shape.get('count')
            if maxCount==None:
                maxCount=int(shape.get('max'))
            else:
                maxCount=int(maxCount)
            sid=int(shape.get('id'))+1
            for _ in range(maxCount):
                shapeIndices.append(sid)
        return shapeIndices

    @staticmethod
    def getShapes(solution): 
        arr=np.array(re.findall(r'x|(-?\d+ -?\d+ -?\d+ -?\d+)', solution.assembly.text))
        shapeIndexList=(np.where( arr!='')[0]).tolist()
        return shapeIndexList

    @staticmethod
    def getShapeNumbers(solution,shapeIndices):
        return [int(shapeIndices[s]) for s in BTtools.getShapes(solution)]

    @staticmethod
    def getIdentities(voxel):    
        return sorted(list(set(re.findall(r'\#\d*|\+\d*',voxel.text))),reverse=True)

    @staticmethod
    def createVoxelbyCoordsTuple(shape,coordsTuple,space='_',mark='#'):
        arr=np.full(shape,space,dtype='|S1')
        arr[tuple(np.array(coords).T)]=mark
        voxel=BTtools.setArray(arr)
        return voxel
    
    @staticmethod
    def setArray(arr):
        voxel=etree.Element('voxel')
        shape=arr.shape
        voxel.set('x',str(shape[X]))
        voxel.set('y',str(shape[Y]))
        voxel.set('z',str(shape[Z]))
        voxel.set('type','0')
        voxel.text=arr.tobytes().decode('utf-8')
        return voxel

    @staticmethod
    def getArray(voxel,mark='x',space='.',var='+'):    
        identities=BTtools.getIdentities(voxel)
        voxelString=voxel.text
        for id in identities:
            if id=='#' or id[0]=='#':
                voxelString=voxelString.replace(id,mark)
            elif id=='+' or id[0]=='+':
                voxelString=voxelString.replace(id,var)
        voxelString=voxelString.replace('_',space)
        z,y,x=int(voxel.get('z')),int(voxel.get('y')),int(voxel.get('x'))
        arr=np.frombuffer(voxelString.encode(),dtype='|S1')
        return np.reshape(arr, ( z, y, x ))

    @staticmethod
    def getAllCoordinates(arr,space=b'.'):        
        coords=np.array(np.where( arr!=space))
        return coords    
    @staticmethod
    def getSecondaryCoordinates(arr):        
        coords=np.array(np.where( (arr==b'+')))
        return coords
    @staticmethod
    def normalizeArray(arr,space=b'.'):
        coords=BTtools.getAllCoordinates(arr,space=space)
        minCoords=[np.min(coords[0]),np.min(coords[1]),np.min(coords[2])]
        normArr = arr[ minCoords[0]:np.max(coords[0]+1) , minCoords[1]:np.max(coords[1]+1) , minCoords[2]:np.max(coords[2]+1) ]  
        return normArr,minCoords


    @staticmethod
    def rotateAll24( arr, out ):
        #njit #( "void(int32[:,:],int32[:],int32[:,:],int32[:,:,:])" )
        z, y, x, zf, yf, xf = 0, 1, 2, 3, 4, 5
        #                    0          1          2          3          4          5          6          7          8          9          10         11         12         13         14         15         16         17         18         19         20         21         22         23     
        k_idx = np.array( [ [0, 1, 2], [1, 3, 2], [3, 4, 2], [4, 0, 2], [2, 1, 3], [2, 3, 4], [2, 4, 0], [2, 0, 1], [3, 1, 5], [4, 3, 5], [0, 4, 5], [1, 0, 5], [5, 1, 0], [5, 3, 1], [5, 4, 3], [5, 0, 4], [0, 2, 4], [1, 2, 0], [3, 2, 1], [4, 2, 3], [0, 5, 1], [1, 5, 3], [3, 5, 4], [4, 5, 0]], dtype = np.int8 )

        tmax = arr[z,-2]
        arr[zf] = - arr[ z ] + tmax
        arr[zf,-2] = tmax
        tmax = arr[y,-2]
        arr[yf] = - arr[ y ] + tmax
        arr[yf,-2] = tmax
        tmax = arr[x,-2]
        arr[xf] = - arr[ x ] + tmax
        arr[xf,-2] = tmax
        for i in range( 24 ):
            out[i,z] = arr[ k_idx[i,z] ]
            out[i,y] = arr[ k_idx[i,y] ]
            out[i,x] = arr[ k_idx[i,x] ]
            
    @staticmethod
    def shift(prop,targetTcoords,targetBound,coords,bound,rot,offset,targetOffset,shft=[0,0,0],axis=0):
        for shft[axis] in range(targetBound[axis]-bound[axis]+1):
            if axis<2:
                BTtools.shift(prop,targetTcoords,targetBound,coords,bound,rot,offset,targetOffset,shft,axis+1)

            Tcoords=tuple( tuple( coord.tolist() ) for coord in coords.T )
            if Tcoords not in prop: # Known problem; Block out double results
                if set(Tcoords).issubset(targetTcoords):
                    btCoords=offset+targetOffset+shft 
                    btString=' '.join([ str(i) for i in  btCoords[::-1]])+' '+str(rot)
                    #print(offset[::-1],btString)
                    #stop()
                    prop[Tcoords]={'bound':bound,'rot':rot,'btString':btString,'targetOffset':targetOffset}     
            newCoords=copy.deepcopy(coords)
            newCoords[axis]+=1
            coords=newCoords

    def getPropositions(self,shapesSelect,target,shift=True,rotations=range(24)):
        propositions={}
        if shift:
            # Get target and secondary coordinaties
            targetArr=self.getArray(self.obj.shapes.voxel[target-1])
            normalizedArr,targetOffset=self.normalizeArray(targetArr)
            targetCoords=self.getAllCoordinates(normalizedArr)
            secondaryCoords=self.getSecondaryCoordinates(normalizedArr)
            targetBound=(targetCoords[0].max(),targetCoords[1].max(),targetCoords[2].max())
            targetTcoords = tuple( tuple( coord.tolist() ) for coord in targetCoords.T )
            secondaryTcoords = tuple( tuple( coord.tolist() ) for coord in secondaryCoords.T )
            propositions[target]={targetTcoords:{'bound':targetBound,'secondary':secondaryTcoords,'btString':'0 0 0 0'}}
        # Rotate and shift
        for shapeNo in shapesSelect:
            shapeArr=self.getArray(self.obj.shapes.voxel[shapeNo-1])
            arr,ofs=self.normalizeArray(shapeArr)
            shapeCoords=self.getAllCoordinates(arr)
            voxelCount=len(shapeCoords[0])
            coords=np.empty( ( 6, voxelCount+2 ), np.int32 )
            coords[0:3,0:-2]=shapeCoords[:,:]
            tmax=[[shapeCoords[0].max()],[shapeCoords[1].max()],[shapeCoords[2].max()]]
            coords[0:3,-2:-1 ] = tmax
            coords[0:3,-1:] = 0
            #print(coords)
            out=np.zeros( ( 24, 3, voxelCount+2 ), np.int32 )
            BTtools.rotateAll24( coords, out )
            propositions[shapeNo]={}
            for rot in rotations: #range(24):
                coords=out[rot][:,:-2]
                coords = coords [ :, coords[2].argsort() ]
                coords = coords [ :, coords[1].argsort( kind = 'mergesort' ) ]
                coords = coords [ :, coords[0].argsort( kind = 'mergesort' ) ]
                TCoords = tuple( tuple( coord ) for coord in coords.T )
                if TCoords not in propositions[shapeNo]:
                    bound=tuple(out[rot][:,-2:-1].T[0])
                    if shift:
                        offset=out[rot][:,-1:].T[0]
                        self.shift(propositions[shapeNo],targetTcoords,targetBound,coords,bound,rot,offset,targetOffset)
                    else:
                        propositions[shapeNo][TCoords]={'bound':bound,'rot':rot}   
                        
        return propositions
    
    def optimizePropositions(self,propositions,shapes,target,verbose=False):
        oPropositions={}
        firstKey =  next(iter(propositions[target]))
        bound=propositions[target][firstKey]['bound']
        targetBound=list(bound).copy()
        targetBoundVoxels=(bound[0]+1)*(bound[1]+1)*(bound[2]+1)
        voxels=len(firstKey)
        for shape in shapes:
            oPropositions[shape]=propositions[shape]
        # Is square or cubic with no holes?
        if 0 in targetBound:
            targetBound.pop(targetBound.index(0))
        if (targetBound.count(targetBound[0]) == len(targetBound) and voxels==targetBoundVoxels):
            maxPropLength=0
            maxPropNo=0
            for k,v in oPropositions.items():
                if len(v)>maxPropLength:
                    maxPropLength=len(v)
                    maxPropNo=k
            anchored={}
            for k,v in oPropositions[maxPropNo].items():
                if v['rot']==0:
                    anchored[k]=v
            if verbose:
                print('Optimize for symmetry: S%s'%maxPropNo,'reduced from',maxPropLength, 'props to',len(anchored))
            oPropositions[maxPropNo]=anchored
        return oPropositions
    
    # Convert propositions to matrix 
    @staticmethod
    def convertToMatrix(propositions,shapesSelect):
        m=[]
        for shape in shapesSelect:
            for k in propositions[shape]:
                m.append([shape]+[ pos for pos in k])
        return m
    
    # get propositions and convert to Burrtool codes and coordinates for given problem
    def getBTcoords(self,problem,ss=None):
        shapes=self.getShapeIndices(problem)
        target=int(problem.result.get('id'))+1
        propositions=self.getPropositions(set(shapes),target)
        targetCoords=[]
        if ss==None:
            targetCoords=next(iter(propositions[target]))
            ss=[(0,0,0),tuple(np.array(propositions[target][targetCoords]['bound'])+1)]
        else:
            for z in range(ss[0][Z],ss[1][Z]):
                for y in range(ss[0][Y],ss[1][Y]):
                    for x in range(ss[0][X],ss[1][X]):
                        targetCoords.append((z,y,x))
        iProp={}
        for s in [target]+shapes:
            iProp[s]={}
            for coords in propositions[s]:
                slicedCoords=[]
                for pos in coords:
                    if pos in targetCoords:
                        slicedCoords.append( ( pos[Z]-ss[0][Z], pos[Y]-ss[0][Y], pos[X]-ss[0][X] ) )
                iProp[s][propositions[s][coords]['btString']]=slicedCoords
        return iProp

    def bt2pcad(self,problemSelect,start,scale=1/1,analyze=True):
        start=start+1
        obj=self.obj
        problem=self.obj.problems.problem[problemSelect-1]
        resultId=int(problem.result.get('id'))
        shapeIndices=[resultId+1]
        shapeCounts=[1]
        if analyze:
            solutions,shapeDict=self.process(problemSelect,[],False)
            for k,v in shapeDict.items():
                if v['frequency']>0:
                    shapeIndices.append(k)
                    shapeCounts.append(v['maxUse'])
        else:
            shapeIndices=self.getShapeIndices(problem)
            uniqueShapes=sorted(list(set(shapeIndices)))
            for shape in uniqueShapes:
                shapeCounts.append(shapeIndices.count(shape))
            shapeIndices=[resultId+1]+list(uniqueShapes)

        fmt='Shapes found: S'+'S'.join(['%d, ']*len(shapeIndices))
        print(fmt[0:-2] % tuple(shapeIndices))
        fmt='Shapes selected: S'+'S'.join(['%d, ']*(len(shapeIndices)-start+1))
        print(fmt[0:-2] % tuple(shapeIndices[start-1:]))
        downScale=int(1/scale)
        voxel=[obj.shapes.voxel[resultId]]
        name=self.obj.problems.problem[problemSelect-1].get('name')
        if name==None:
            name=''
        voxel[0].set('name','%s (P%d target shape)'%(name,problemSelect))
        for i,id in enumerate(shapeIndices[1:]):
            problem.shapes.shape[i].set('id',str(i))
            voxel.append(obj.shapes.voxel[id-1])
        problem.result.set('id','0')
        obj.problems.problem=[problem]
        obj.shapes.voxel=voxel
        pcad='burr_plate(\n['
        for no,voxel in enumerate(obj.shapes.voxel[start-1:]):
            name=voxel.get('name')
            if name==None:
                name=''
            else:
                name='('+name+')'
            arr=BTtools.getArray(voxel)
            normArr,ofs=BTtools.normalizeArray(arr)
            scaledArr=normArr[0::downScale,0::downScale,0::downScale].copy()
            if analyze:
                shapeCount=shapeCounts[no+start-1]
            else:
                shapeCount=1
            for i in range(shapeCount):
                num = '-'+str(i+1) if shapeCount>1 else ''
                pcad+='\t'+str(['|'.join([ a.tobytes().decode('utf-8') for a in scaledArr[i,:] ]) for i in range(len(scaledArr))])+',\t// S'+str(shapeIndices[no+start-1])+num+' '+name+'\n'
        pcad+=']);\n'
        pcad=pcad.replace("'",'"');
        return pcad

    @staticmethod
    def floodFill( arr, pos, oldVal, newVal ):
        shape = arr.shape
        if arr[pos] == oldVal:
            arr[pos] = newVal
            if pos[Z] > 0:
                BTtools.floodFill( arr, ( pos[Z]-1, pos[Y],   pos[X] ),   oldVal, newVal )
            if pos[Z] < (shape[Z]-1):
                BTtools.floodFill( arr, ( pos[Z]+1, pos[Y],   pos[X] ),   oldVal, newVal )
            if pos[Y] > 0:
                BTtools.floodFill( arr, ( pos[Z],   pos[Y]-1, pos[X] ),   oldVal, newVal )
            if pos[Y] < (shape[Y]-1):
                BTtools.floodFill( arr, ( pos[Z],   pos[Y]+1, pos[X] ),   oldVal, newVal )
            if pos[X] > 0:
                BTtools.floodFill( arr, ( pos[Z],   pos[Y],   pos[X]-1 ), oldVal, newVal )
            if pos[X] < (shape[X]-1):
                BTtools.floodFill( arr, ( pos[Z],   pos[Y],   pos[X]+1 ), oldVal, newVal )
    def write(self,extension=''):
        global files
        newXml = etree.tostring(self.obj)
        p=Path(self.filename)
        if extension != '':
            extension=','+extension
        dst_filename=p.parent.joinpath(p.stem+extension+'.xmpuzzle')
        print('Writing: ',dst_filename)
        with gzip.open(dst_filename, 'wb') as f:
            f.write(newXml)
            f.close()
        if IN_COLAB:
            if self.filename=='' or self.filename==None:
                self.filename='tmp.xmpuzzle'
            files.download(dst_filename)
